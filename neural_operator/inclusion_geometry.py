"""Conforming OCC meshes for explicitly specified circular inclusions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from uuid import uuid4

import numpy as np


def generate_inclusion_mesh(path, circles, *, h=.1, n_coarse=4):
    """Mesh filled circles in [-2,2]^2, assigning ordinal material tags.

    ``circles`` contains (center_x, center_y, radius), in tag order 2..N+1.
    The background has tag 1. Boundary tags are left=11, right=12, top=13,
    bottom=14. A neighboring ``.mesh_info.json`` records geometry, meshing
    parameters, verified CAD areas and the mesh checksum. Empty circles are
    supported. Circles must lie strictly inside the square and not touch.
    """
    circles = np.asarray(circles, dtype=np.float64)
    if circles.shape == (0,):
        circles = circles.reshape(0,3)
    if (circles.ndim != 2 or circles.shape[1] != 3 or
            not np.isfinite(circles).all() or np.any(circles[:,2] <= 0)):
        raise ValueError("circles must be finite [count,3] with positive radii")
    if np.any(np.abs(circles[:,:2])+circles[:,2,None] >= 2):
        raise ValueError("circles must lie strictly inside [-2,2]^2")
    for index, circle in enumerate(circles):
        if np.any(np.linalg.norm(circles[:index,:2]-circle[:2],axis=1) <= circles[:index,2]+circle[2]):
            raise ValueError("circles must be disjoint and must not touch")
    if not np.isfinite(h) or h <= 0:
        raise ValueError("h must be finite and positive")
    if isinstance(n_coarse,bool) or not np.isfinite(n_coarse) or int(n_coarse) != n_coarse or n_coarse < 1:
        raise ValueError("n_coarse must be a positive integer")
    import gmsh
    from gmsfem.geometry import _interior_point

    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True,exist_ok=True)
    own_session = not gmsh.isInitialized()
    if own_session:
        gmsh.initialize()
    previous = gmsh.model.getCurrent()
    minimum_radius = float(circles[:,2].min()) if len(circles) else None
    options = {"General.Verbosity":1,
               "Mesh.MeshSizeMin":min(h/10,2*np.pi*minimum_radius/32) if minimum_radius is not None else h/10,
               "Mesh.MeshSizeMax":float(h), "Mesh.MeshSizeFromCurvature":32,
               "Mesh.MinimumCirclePoints":32, "Mesh.MeshSizeExtendFromBoundary":0,
               "Mesh.ElementOrder":1, "Mesh.MshFileVersion":4.1, "Mesh.Binary":0}
    previous_options = {key:gmsh.option.getNumber(key) for key in options}
    try:
        for key,value in options.items():
            gmsh.option.setNumber(key,value)
        gmsh.model.add("inclusion_geometry_"+uuid4().hex)
        occ = gmsh.model.occ
        step = 4/int(n_coarse)
        rectangles = [(2,occ.addRectangle(-2+i*step,-2+j*step,0,step,step))
                      for j in range(int(n_coarse)) for i in range(int(n_coarse))]
        disks = [(2,occ.addDisk(float(x),float(y),0,float(radius),float(radius)))
                 for x,y,radius in circles]
        if disks:
            occ.fragment(rectangles,disks)
        elif len(rectangles) > 1:
            occ.fragment(rectangles[:1],rectangles[1:])
        occ.synchronize()
        groups = {tag:[] for tag in range(1,len(circles)+2)}
        areas = {tag:0. for tag in groups}
        for _,surface in gmsh.model.getEntities(2):
            point = _interior_point(gmsh,surface)[:2]
            matches = np.flatnonzero(np.sum((circles[:,:2]-point)**2,axis=1) < circles[:,2]**2)
            tag = int(matches[0])+2 if len(matches) else 1
            groups[tag].append(surface)
            areas[tag] += float(occ.getMass(2,surface))
        for tag,surfaces in groups.items():
            if not surfaces:
                raise RuntimeError(f"CAD material tag {tag} is empty")
            gmsh.model.addPhysicalGroup(2,surfaces,tag)
            gmsh.model.setPhysicalName(2,tag,"background" if tag == 1 else f"inclusion_{tag-2}")
            expected = float(np.pi*circles[tag-2,2]**2) if tag > 1 else float(16-np.sum(np.pi*circles[:,2]**2))
            if not np.isclose(areas[tag],expected,rtol=1e-7,atol=1e-10):
                raise RuntimeError(f"CAD area for tag {tag} is {areas[tag]}, expected {expected}")
        if not np.isclose(sum(areas.values()),16.,rtol=0,atol=1e-9):
            raise RuntimeError("Total CAD area must equal 16")
        eps = 1e-6
        boxes = {11:(-2-eps,-2-eps,-eps,-2+eps,2+eps,eps),
                 12:(2-eps,-2-eps,-eps,2+eps,2+eps,eps),
                 13:(-2-eps,2-eps,-eps,2+eps,2+eps,eps),
                 14:(-2-eps,-2-eps,-eps,2+eps,-2+eps,eps)}
        names = {11:"left",12:"right",13:"top",14:"bottom"}
        for tag,box in boxes.items():
            curves = [entity for _,entity in gmsh.model.getEntitiesInBoundingBox(*box,dim=1)]
            if not curves:
                raise RuntimeError(f"Missing CAD boundary {names[tag]}")
            gmsh.model.addPhysicalGroup(1,curves,tag)
            gmsh.model.setPhysicalName(1,tag,names[tag])
        gmsh.model.mesh.generate(2)
        gmsh.write(str(path))
        info = {"generator":"explicit_circular_inclusions", "h":float(h), "n_coarse":int(n_coarse),
                "circles":circles.tolist(), "circle_tags":list(range(2,len(circles)+2)),
                "background_tag":1, "boundary_tags":names,
                "material_areas":areas, "cad_total_area":sum(areas.values()),
                "material_surfaces":groups, "gmsh_version":gmsh.__version__,
                "msh_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                "generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        path.with_suffix(".mesh_info.json").write_text(json.dumps(info,indent=2)+"\n")
    finally:
        if own_session:
            gmsh.finalize()
        else:
            gmsh.model.remove()
            for key,value in previous_options.items():
                gmsh.option.setNumber(key,value)
            if previous:
                gmsh.model.setCurrent(previous)
    return path
