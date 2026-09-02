import os;
import pymeshlab;

input_dir = "./meshes";
output_dir = "./meshes_low";
ratio = 0.01;

os.makedirs(output_dir, exist_ok=True);

for filename in os.listdir(input_dir):
    if not filename.lower().endswith(".stl"):
        continue;

    input_path = os.path.join(input_dir, filename);
    output_path = os.path.join(output_dir, filename);

    ms = pymeshlab.MeshSet();
    ms.load_new_mesh(input_path);

    old_faces = ms.current_mesh().face_number();

    ms.meshing_decimation_quadric_edge_collapse(
        targetperc=ratio,
        preservenormal=True,
        preserveboundary=True
    );

    new_faces = ms.current_mesh().face_number();

    ms.save_current_mesh(output_path);

    print(f"{filename}: {old_faces} -> {new_faces}, ratio={new_faces / old_faces:.3f}");