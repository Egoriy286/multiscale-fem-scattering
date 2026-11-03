// Filled inclusions in [-2,2]^2. Parameters may be supplied with -setnumber.
SetFactory("OpenCASCADE");
If (!Exists(h_default))
  h_default = 0.04;
EndIf
DefineConstant[
  n_coarse = {10, Min 1, Step 1, Name "GMSFEM/n_coarse"},
  h = {h_default, Min 0.0001, Name "GMSFEM/h"}
];

// Original image coordinates: x = CX*0.004-2, y = 2-CY*0.004.
cx[] = {615,450,525,625,350,450,550,675,350,525,450,600,
        675,525,425,475,550,625,350,425,635,500,575,335};
cy[] = {285,325,325,325,375,400,400,400,475,475,500,500,
        500,525,575,575,600,600,625,650,665,675,675,715};
ss[] = {10,25,25,25,25,50,50,25,25,25,50,50,
        25,25,25,25,50,25,25,25,10,25,25,10};

cells[] = {};
For j In {0:n_coarse-1}
  For i In {0:n_coarse-1}
    s = news;
    Rectangle(s) = {-2+4*i/n_coarse, -2+4*j/n_coarse, 0,
                    4/n_coarse, 4/n_coarse};
    cells[] += {s};
  EndFor
EndFor
inclusions[] = {};
For k In {0:23}
  s = news;
  Disk(s) = {cx[k]*0.004-2, 2-cy[k]*0.004, 0, ss[k]*0.002};
  inclusions[] += {s};
EndFor
fragments[] = BooleanFragments{ Surface{cells[]}; Delete; }
                             { Surface{inclusions[]}; Delete; };

// Materials are assigned by geometry.py using interior points of these
// trimmed OCC faces. Bounding boxes alone cannot distinguish annuli.
eps = 1.e-6;
left[] = Curve In BoundingBox{-2-eps,-2-eps,-eps,-2+eps,2+eps,eps};
right[] = Curve In BoundingBox{2-eps,-2-eps,-eps,2+eps,2+eps,eps};
top[] = Curve In BoundingBox{-2-eps,2-eps,-eps,2+eps,2+eps,eps};
bottom[] = Curve In BoundingBox{-2-eps,-2-eps,-eps,2+eps,-2+eps,eps};
Physical Curve("left", 11) = {left[]};
Physical Curve("right", 12) = {right[]};
Physical Curve("top", 13) = {top[]};
Physical Curve("bottom", 14) = {bottom[]};
// h controls the background, not a lower bound on curved interfaces.
// 32 points per complete circle keep the linear polygon area error below 1%,
// including the three smallest inclusions (radius 0.02).
Mesh.MeshSizeMin = Min(h/10, 2*Pi*0.02/32);
Mesh.MeshSizeMax = h;
Mesh.MeshSizeFromCurvature = 32;
Mesh.MinimumCirclePoints = 32;
Mesh.MeshSizeExtendFromBoundary = 1;
Mesh.ElementOrder = 1;
Mesh.MshFileVersion = 4.1;
Mesh.Binary = 0;
