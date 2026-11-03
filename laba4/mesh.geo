// Квадрат 1x1 из n x n блоков с треугольной сеткой шага примерно h.
SetFactory("OpenCASCADE");
n = 10;
fine = 12;
h = 1/(n*fine);

For j In {0:n-1}
  For i In {0:n-1}
    Rectangle(news) = {i/n, j/n, 0, 1/n, 1/n};
  EndFor
EndFor

// Соседние блоки используют общие точки и линии.
Coherence;
Mesh.MeshSizeMin = h;
Mesh.MeshSizeMax = h;
Mesh.MshFileVersion = 2.2;
Mesh.SaveAll = 1;
