import json
import sys
import os
import time
from pathlib import Path
from typing import List, Dict, Tuple, Set
import numpy as np
import open3d as o3d
from dataclasses import dataclass
import cProfile
import pstats
import io
import argparse

try:
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_VERTEX
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopoDS import topods
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.BRep import BRep_Tool, BRep_Builder
    from OCC.Core.TopoDS import TopoDS_Compound
    from OCC.Core.TopLoc import TopLoc_Location
    PYTHONOCC_AVAILABLE = True
    print("✓ pythonocc-core imports successful")
except ImportError as e:
    print(f"✗ pythonocc-core import failed: {e}")
    PYTHONOCC_AVAILABLE = False


@dataclass
class ViewInfo:
    name: str
    direction: np.ndarray
    up_vector: np.ndarray = None


class RealGeometryStepAnalyzer:
    
    def __init__(self):
        self.canonical_views = {
            'ViewTop': ViewInfo('ViewTop', np.array([0, 0, -1]), np.array([0, 1, 0])),
            'ViewBottom': ViewInfo('ViewBottom', np.array([0, 0, 1]), np.array([0, 1, 0])),
            'ViewFront': ViewInfo('ViewFront', np.array([0, -1, 0]), np.array([0, 0, 1])),
            'ViewRear': ViewInfo('ViewRear', np.array([0, 1, 0]), np.array([0, 0, 1])),
            'ViewRight': ViewInfo('ViewRight', np.array([1, 0, 0]), np.array([0, 0, 1])),
            'ViewLeft': ViewInfo('ViewLeft', np.array([-1, 0, 0]), np.array([0, 0, 1]))
        }
        
        # Angle threshold for face visibility set to 60.
        self.angle_threshold = np.cos(np.radians(60))
        
    def load_step_file(self, step_file: str):
        """Load a STEP file using pythonocc-core, handling multiple shapes."""
        print(f"Loading STEP file: {step_file}")
        
        step_reader = STEPControl_Reader()
        status = step_reader.ReadFile(step_file)
        
        if status != 1:  
            raise ValueError(f"Error reading STEP file: {step_file}")
            
        step_reader.TransferRoots()

        builder = BRep_Builder()
        compound = TopoDS_Compound()
        builder.MakeCompound(compound)
        
        num_shapes = step_reader.NbShapes()
        if num_shapes == 0:
            raise ValueError("No shapes found in STEP file.")

        for i in range(1, num_shapes + 1):
            shape_to_add = step_reader.Shape(i)
            builder.Add(compound, shape_to_add)

        shape = compound
        
        if shape.IsNull():
            raise ValueError("Failed to load shape from STEP file")
            
        print(f"Successfully loaded STEP file with {num_shapes} root shape(s). Combined shape type: {shape.ShapeType()}")
        return shape
    
    def extract_real_triangulation(self, shape, linear_deflection: float = 0.01):
        """Extract real triangulated mesh from STEP shape"""
        print("Extracting real triangulated mesh from STEP shape...")
        
        mesher = BRepMesh_IncrementalMesh(shape, linear_deflection)
        mesher.Perform()
        
        if not mesher.IsDone():
            raise ValueError("Tessellation failed")
        
        all_vertices = []
        all_triangles = []
        face_triangle_map = {}
        current_triangle_idx = 0
        
        face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
        face_id = 0
        
        while face_explorer.More():
            face = topods.Face(face_explorer.Current())
            
            location = TopLoc_Location()
            triangulation = BRep_Tool.Triangulation(face, location)
            
            if triangulation is not None:
                print(f"Processing face {face_id} with {triangulation.NbTriangles()} triangles")
                
                transform = location.Transformation()
                
                face_vertices = []
                
                for i in range(1, triangulation.NbNodes() + 1):  # 1-based indexing
                    pnt = triangulation.Node(i)
                    # transformation
                    if not location.IsIdentity():
                        pnt.Transform(transform)
                    face_vertices.append([pnt.X(), pnt.Y(), pnt.Z()])
                
                face_triangles = []
                
                for i in range(1, triangulation.NbTriangles() + 1):  # 1-based indexing
                    triangle = triangulation.Triangle(i)
                    n1, n2, n3 = triangle.Get()
                    
                    tri = [
                        n1 - 1 + len(all_vertices),
                        n2 - 1 + len(all_vertices), 
                        n3 - 1 + len(all_vertices)
                    ]
                    face_triangles.append(tri)
                
                all_vertices.extend(face_vertices)
                all_triangles.extend(face_triangles)
                
                # Map face to triangle indices
                face_triangle_indices = list(range(current_triangle_idx, 
                                                 current_triangle_idx + len(face_triangles)))
                face_triangle_map[face_id] = face_triangle_indices
                current_triangle_idx += len(face_triangles)
                
                face_id += 1
            
            face_explorer.Next()
        
        print(f"Extracted real mesh: {len(all_vertices)} vertices, {len(all_triangles)} triangles")
        print(f"Found {len(face_triangle_map)} faces with triangulation")
        
        if len(all_vertices) == 0 or len(all_triangles) == 0:
            print("Warning: No triangulation found, falling back to bounding box")
            return self._create_fallback_mesh(shape)
        
        return np.array(all_vertices), np.array(all_triangles), face_triangle_map
    
    def _create_fallback_mesh(self, shape):
        """Create fallback mesh from bounding box if triangulation fails"""
        vertices = []
        vertex_explorer = TopExp_Explorer(shape, TopAbs_VERTEX)
        
        while vertex_explorer.More():
            vertex = vertex_explorer.Current()
            pnt = BRep_Tool.Pnt(topods.Vertex(vertex))
            vertices.append([pnt.X(), pnt.Y(), pnt.Z()])
            vertex_explorer.Next()
        
        if len(vertices) > 0:
            vertices_array = np.array(vertices)
            min_coords = np.min(vertices_array, axis=0)
            max_coords = np.max(vertices_array, axis=0)
            print(f"Using bounding box fallback: min={min_coords}, max={max_coords}")
            return self._create_box_mesh(min_coords, max_coords)
        else:
            print("Using unit cube fallback")
            return self._create_box_mesh([-1, -1, -1], [1, 1, 1])
    
    def _create_box_mesh(self, min_coords, max_coords):
        """Create a simple box mesh from min/max coordinates"""
        x_min, y_min, z_min = min_coords
        x_max, y_max, z_max = max_coords
        
        vertices = [
            [x_min, y_min, z_min], [x_max, y_min, z_min], [x_max, y_max, z_min], [x_min, y_max, z_min],  # bottom
            [x_min, y_min, z_max], [x_max, y_min, z_max], [x_max, y_max, z_max], [x_min, y_max, z_max]   # top
        ]
        
        # Box triangles (2 triangles per face, 6 faces)
        triangles = [
            # Bottom face (z_min)
            [0, 1, 2], [0, 2, 3],
            # Top face (z_max)
            [4, 6, 5], [4, 7, 6],
            # Front face (y_min)
            [0, 4, 5], [0, 5, 1],
            # Back face (y_max)
            [2, 6, 7], [2, 7, 3],
            # Left face (x_min)
            [0, 3, 7], [0, 7, 4],
            # Right face (x_max)
            [1, 5, 6], [1, 6, 2]
        ]
        
        # Create face mapping
        face_triangle_map = {
            0: [0, 1],    # Bottom
            1: [2, 3],    # Top
            2: [4, 5],    # Front
            3: [6, 7],    # Back
            4: [8, 9],    # Left
            5: [10, 11]   # Right
        }
        
        return vertices, triangles, face_triangle_map
    
    def create_open3d_mesh(self, vertices: np.ndarray, triangles: np.ndarray) -> o3d.geometry.TriangleMesh:
        """Create an Open3D mesh from vertices and triangles"""
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        mesh.compute_triangle_normals()
        mesh.compute_vertex_normals()
        return mesh
    
    def compute_face_visibility(self, mesh: o3d.geometry.TriangleMesh, 
                               face_triangle_map: Dict[int, List[int]],
                               view_direction: np.ndarray) -> Dict[int, bool]:
        """Compute which faces are visible from a given view direction"""
        face_visibility = {}
        triangle_normals = np.asarray(mesh.triangle_normals)
        
        for face_id, triangle_indices in face_triangle_map.items():
            face_visible = False
            
            for tri_idx in triangle_indices:
                if tri_idx >= len(triangle_normals):
                    continue
                    
                # Check orientation - face normal should align with view direction
                normal = triangle_normals[tri_idx]
                dot_product = np.dot(normal, -view_direction)
                
                if dot_product > self.angle_threshold:
                    face_visible = True
                    break
            
            face_visibility[face_id] = face_visible
        
        return face_visibility
    
    def greedy_view_selection(self, critical_faces: Set[int], 
                             face_coverage: Dict[str, Dict[int, bool]]) -> List[str]:
        """Select minimal set of views that cover all critical faces"""
        uncovered_faces = set(critical_faces)
        selected_views = []
        
        while uncovered_faces and len(selected_views) < 6:
            best_view = None
            best_coverage = 0
            
            # Find view that covers most uncovered faces
            for view_name, visibility in face_coverage.items():
                if view_name in selected_views:
                    continue
                    
                coverage_count = sum(1 for face_id in uncovered_faces 
                                   if face_id in visibility and visibility[face_id])
                
                if coverage_count > best_coverage:
                    best_coverage = coverage_count
                    best_view = view_name
            
            if best_view is None or best_coverage == 0:
                break
                
            selected_views.append(best_view)
            
            newly_covered = set()
            for face_id in uncovered_faces:
                if (face_id in face_coverage[best_view] and 
                    face_coverage[best_view][face_id]):
                    newly_covered.add(face_id)
            
            uncovered_faces -= newly_covered
            print(f"Selected {best_view}, covers {len(newly_covered)} faces, "
                  f"{len(uncovered_faces)} faces remaining")
        
        return selected_views
    
    def render_view(self, mesh: o3d.geometry.TriangleMesh, 
                   view_info: ViewInfo,
                   covered_faces: Set[int],
                   face_triangle_map: Dict[int, List[int]],
                   output_path: str):
        """Render a view with covered faces highlighted"""
        print(f"Rendering view: {view_info.name}")
        
        import copy
        colored_mesh = copy.deepcopy(mesh)
        
        num_triangles = len(mesh.triangles)
        colors = np.full((num_triangles, 3), [0.7, 0.7, 0.7])  # Gray
        
        # Color covered faces in red
        for face_id in covered_faces:
            if face_id in face_triangle_map:
                for tri_idx in face_triangle_map[face_id]:
                    if tri_idx < num_triangles:
                        colors[tri_idx] = [1.0, 0.2, 0.2]  # Red
        
        try:
            colored_mesh.vertex_colors = o3d.utility.Vector3dVector([])
            vertex_colors = np.full((len(mesh.vertices), 3), [0.7, 0.7, 0.7])
            triangles_array = np.asarray(mesh.triangles)
            
            for face_id in covered_faces:
                if face_id in face_triangle_map:
                    for tri_idx in face_triangle_map[face_id]:
                        if tri_idx < len(triangles_array):
                            for vertex_idx in triangles_array[tri_idx]:
                                vertex_colors[vertex_idx] = [1.0, 0.2, 0.2]  # Red
            
            colored_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
        except:
            colored_mesh.paint_uniform_color([0.7, 0.7, 0.7])
        
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=800, height=600)
        vis.add_geometry(colored_mesh)
        
        view_control = vis.get_view_control()
        
        # camera position
        bbox = mesh.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        extent = bbox.get_extent()
        distance = np.linalg.norm(extent) * 2
        
        camera_pos = center + view_info.direction * distance
        
        # camera parameters
        view_control.set_front(view_info.direction)
        view_control.set_up(view_info.up_vector)
        view_control.set_lookat(center)
        
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(output_path)
        vis.destroy_window()
        
        print(f"Saved render to: {output_path}")
    
    def analyze_step_file(self, step_file: str, critical_faces_file: str, output_dir: str):
        """Main analysis function with real geometry extraction"""
        start_time = time.time()
        
        os.makedirs(output_dir, exist_ok=True)
        renders_dir = os.path.join(output_dir, 'renders')
        os.makedirs(renders_dir, exist_ok=True)
        
        with open(critical_faces_file, 'r') as f:
            critical_faces_data = json.load(f)
        
        # The loaded IDs are already 0-based, use them directly
        critical_face_ids = critical_faces_data.get('critical_face_ids', [])
        critical_faces = set(critical_face_ids)
        
        print(f"Loaded {len(critical_face_ids)} critical faces. Using 0-indexed: {critical_faces}")
        
        if not PYTHONOCC_AVAILABLE:
            raise ImportError("pythonocc-core not available for real STEP analysis")
        
        #tessellation of mesh
        shape = self.load_step_file(step_file)
        vertices, triangles, face_triangle_map = self.extract_real_triangulation(shape)
        
        mesh = self.create_open3d_mesh(vertices, triangles)
        
        print(f"Created mesh with {len(vertices)} vertices and {len(triangles)} triangles")
        print(f"Face mapping: {len(face_triangle_map)} faces")
        
        face_coverage = {}
        for view_name, view_info in self.canonical_views.items():
            print(f"Computing visibility for {view_name}")
            visibility = self.compute_face_visibility(mesh, face_triangle_map, view_info.direction)
            face_coverage[view_name] = visibility
            visible_faces = [f for f, v in visibility.items() if v]
            print(f"  {view_name} sees faces: {visible_faces}")
        
        # Select minimal views
        selected_views = self.greedy_view_selection(critical_faces, face_coverage)
        
        views_output = []
        for view_name in selected_views:
            covered_face_ids = [face_id for face_id in critical_faces 
                              if (face_id in face_coverage[view_name] and 
                                  face_coverage[view_name][face_id])]
            
            views_output.append({
                'view': view_name,
                'covered_faces': covered_face_ids
            })
            
            render_path = os.path.join(renders_dir, f"{view_name}.png")
            self.render_view(mesh, self.canonical_views[view_name], 
                           set(covered_face_ids), face_triangle_map, render_path)
        
        views_json_path = os.path.join(output_dir, 'views.json')
        with open(views_json_path, 'w') as f:
            json.dump(views_output, f, indent=2)
        
        all_covered = set()
        for view_data in views_output:
            all_covered.update(view_data['covered_faces'])
        
        coverage_percentage = len(all_covered) / len(critical_faces) * 100 if critical_faces else 100
        
        end_time = time.time()
        runtime = end_time - start_time
        
        print(f"\n=== ANALYSIS COMPLETE ===")
        print(f"Runtime: {runtime:.2f} seconds")
        print(f"Selected views: {len(selected_views)}")
        print(f"Critical faces covered: {len(all_covered)}/{len(critical_faces)} ({coverage_percentage:.1f}%)")
        print(f"Uncovered faces: {critical_faces - all_covered}")
        print(f"Output saved to: {output_dir}")
        
        return {
            'runtime': runtime,
            'views_count': len(selected_views),
            'coverage_percentage': coverage_percentage,
            'selected_views': selected_views
        }


def main():
    parser = argparse.ArgumentParser(description="3D to 2D CAD Drawing View Optimizer")
    parser.add_argument("step_file", help="Path to the STEP file.")
    parser.add_argument("critical_faces_json", help="Path to the JSON file with critical face IDs.")
    parser.add_argument("output_dir", help="Directory to save the output files.")
    parser.add_argument("--profile", action="store_true", help="Enable profiling and print a summary of top functions.")

    args = parser.parse_args()

    if args.profile:
        pr = cProfile.Profile()
        pr.enable()

    try:
        analyzer = RealGeometryStepAnalyzer()
        results = analyzer.analyze_step_file(args.step_file, args.critical_faces_json, args.output_dir)
        print(f"Analysis completed successfully!")
    except Exception as e:
        print(f"Failed to start analysis. Error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.profile:
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.strip_dirs().print_stats(15)  # Show top 15 functions

        profile_summary = s.getvalue()
        
        print("\n--- Profiling Summary (Top 15 Functions by Cumulative Time) ---")
        print(profile_summary)

        log_file = Path(args.output_dir) / "analysis.log"
        if log_file.exists():
            with open(log_file, "a") as f:
                f.write("\n--- Profiling Summary (Top 15 Functions by Cumulative Time) ---\n")
                f.write(profile_summary)


if __name__ == "__main__":
    main() 