from glass_engine import *
from glass_engine.Geometries import *

scene, camera, light, floor = SceneRoam()

geoms = \
[
    Sphere(radius=0.5)    , Cone(radius=0.5)        , Cylinder(radius=0.5),
    Box(Lx=0.7)           , Prism(radius=0.5)       , Pyramid(radius=0.5),
    Octahedron(radius=0.5), Dodecahedron(radius=0.5), Icosahedron(radius=0.5)
]

for i in range(len(geoms)):
    geoms[i].position.x = 2*(i % 3 - 1)
    geoms[i].position.y = 2*(1 - i // 3)
    geoms[i].position.z -= geoms[i].z_min
    scene.add(geoms[i])

camera.screen.show()