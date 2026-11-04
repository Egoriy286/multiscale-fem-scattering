"""Actual OCC geometry and physical mesh-tag regression tests."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from matplotlib.tri import Triangulation

from neural_operator.inclusion_geometry import generate_inclusion_mesh


class InclusionGeometryTests(unittest.TestCase):
    def setUp(self):
        try:
            import gmsh
        except ImportError:
            self.skipTest("gmsh is required for real geometry tests")
        self.gmsh = gmsh
        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity",0)
        self.temporary = tempfile.TemporaryDirectory()

    def tearDown(self):
        if hasattr(self,"gmsh") and self.gmsh.isInitialized():
            self.gmsh.finalize()
        if hasattr(self,"temporary"):
            self.temporary.cleanup()

    def test_shifted_circles_conforming_tags_areas_and_centers(self):
        circles = np.array([[0.,0.,.22], [.83,.31,.19], [-1.12,.81,.13]])
        path = Path(self.temporary.name)/"shifted.msh"
        result = generate_inclusion_mesh(path,circles,h=.3,n_coarse=4)
        self.assertEqual(result,path.resolve())
        info = json.loads(path.with_suffix(".mesh_info.json").read_text())
        np.testing.assert_array_equal(info["circles"],circles)
        self.assertEqual(info["circle_tags"],[2,3,4])
        self.assertEqual(info["n_coarse"],4)
        self.assertEqual(info["msh_sha256"],hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertAlmostEqual(info["cad_total_area"],16.,places=9)
        for index,(_,_,radius) in enumerate(circles):
            self.assertAlmostEqual(info["material_areas"][str(index+2)],np.pi*radius**2,places=9)
        # The first circle crosses four coarse cells, yet keeps one material tag.
        self.assertGreaterEqual(len(info["material_surfaces"]["2"]),4)
        gmsh = self.gmsh
        gmsh.open(str(path))
        self.assertEqual({tag for _,tag in gmsh.model.getPhysicalGroups(2)},{1,2,3,4})
        self.assertEqual({tag for _,tag in gmsh.model.getPhysicalGroups(1)},{11,12,13,14})
        node_tags,coordinates,_ = gmsh.model.mesh.getNodes()
        coordinates = coordinates.reshape(-1,3)[:,:2]
        node_index = {int(tag):i for i,tag in enumerate(node_tags)}
        triangles,tags = [],[]
        for _,tag in gmsh.model.getPhysicalGroups(2):
            for surface in gmsh.model.getEntitiesForPhysicalGroup(2,tag):
                _,nodes = gmsh.model.mesh.getElementsByType(2,int(surface))
                cells = np.array([node_index[int(node)] for node in nodes]).reshape(-1,3)
                triangles.extend(cells)
                tags.extend([tag]*len(cells))
        triangles,tags = np.asarray(triangles),np.asarray(tags)
        xy = coordinates[triangles]
        edges = xy[:,1:]-xy[:,:1]
        areas = abs(edges[:,0,0]*edges[:,1,1]-edges[:,0,1]*edges[:,1,0])/2
        self.assertAlmostEqual(float(areas.sum()),16.,places=9)
        for index,radius in enumerate(circles[:,2]):
            np.testing.assert_allclose(areas[tags == index+2].sum(),np.pi*radius**2,rtol=.012)
        locator = Triangulation(coordinates[:,0],coordinates[:,1],triangles).get_trifinder()
        containing = locator(circles[:,0],circles[:,1])
        self.assertTrue(np.all(containing >= 0))
        np.testing.assert_array_equal(tags[containing],np.arange(2,5))

    def test_empty_geometry_and_existing_model_are_preserved(self):
        gmsh = self.gmsh
        gmsh.model.add("unrelated_geometry")
        rectangle = gmsh.model.occ.addRectangle(0,0,0,1,1)
        gmsh.model.occ.synchronize()
        previous_h = gmsh.option.getNumber("Mesh.MeshSizeMax")
        path = generate_inclusion_mesh(Path(self.temporary.name)/"empty.msh",[],h=.5,n_coarse=2)
        self.assertEqual(gmsh.model.getCurrent(),"unrelated_geometry")
        self.assertAlmostEqual(gmsh.model.occ.getMass(2,rectangle),1.)
        self.assertEqual(gmsh.option.getNumber("Mesh.MeshSizeMax"),previous_h)
        info = json.loads(path.with_suffix(".mesh_info.json").read_text())
        self.assertEqual(info["circles"],[])
        self.assertEqual(info["material_areas"],{"1":16.})

    def test_invalid_circle_geometry_and_parameters(self):
        path = Path(self.temporary.name)/"invalid.msh"
        invalid = ([[0.,0.,-.1]], [[np.nan,0.,.1]], [[1.9,0.,.2]],
                   [[0.,0.,.2],[.3,0.,.2]], [[0.,0.,.2],[.4,0.,.2]], [[0.,1.]])
        for circles in invalid:
            with self.subTest(circles=circles),self.assertRaises(ValueError):
                generate_inclusion_mesh(path,circles)
        for options in ({"h":0},{"h":np.inf},{"n_coarse":0},{"n_coarse":2.5}):
            with self.subTest(options=options),self.assertRaises(ValueError):
                generate_inclusion_mesh(path,[[0.,0.,.2]],**options)
        self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
