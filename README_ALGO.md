# How 2D view generator works:

### Step 1: Understanding the 3D Model

It loads the step files using pythonocc-core and then tessalates it to convert the smooth surface into collection of traingles for the open3D to work with it. Also reocrds info on which faces belong to which surface.

### Step 2: Defining the "Cameras"

The program simulates the standard views used in technical drawings. It sets up six virtual "cameras" around the model: Top bottom, front, rear, right and left.

### Step 3: Checking What Each Camera Sees

Next, the program figures out which faces are visible from each of the six camera angles.

A face is considered "visible" if its surface is generally pointing towards the camera. The program does this by looking at the triangles for each face. Every triangle has a "normal" vector—an imaginary arrow pointing straight out from its surface.
If a triangle's normal vector is pointed towards the camera, it's visible. If it's pointed away, it's hidden. The program creates a master list that records, for each of the six views, every single face that is visible from that angle.

### Step 4: The "Greedy" Algorithm for Choosing the Best Views

1.  The program begins with the full list of critical faces that need to be seen.
2.  It examines all six views and asks: "Which single view reveals the highest number of *not-yet-seen* critical faces?"
3.  The view that provides the most coverage is selected and added to the final list of required drawings.
4.  The faces covered by the selected view are now considered "seen" and are removed from the list of faces it needs to worry about.
5.  The process repeats. It again finds the view that covers the most *remaining* faces, adds it to the list, and updates the list of unseen faces.

This cycle continues until every critical face is accounted for, or until adding more views provides no new coverage.
 