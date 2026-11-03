LX1 = -2.0;
LY1 = -2.0;
LX2 = 2.0;
LY2 = 2.0;
ls = 0.05;
ls2 = 0.05;
Point(1) = {LX1, LY1, 0, ls};
Point(2) = {LX1, LY2, 0, ls};
Point(3) = {LX2, LY2, 0, ls};
Point(4) = {LX2, LY1, 0, ls};

Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,1};

Line Loop(1) = {1,2,3,4};
loops[] = {};
Function circle
    CX = CX / 1000 * 4 - 2.0;
    CY = -CY / 1000 * 4 + 2.0;
    s = S / 1000 * 2;
    s = s;
    p = newp; Point(p)   = {CX,CY,0,  ls2};
    p1 = newp; Point(p1) = {CX-s,CY,0,ls2};
    p2 = newp; Point(p2) = {CX,CY+s,0,ls2};
    p3 = newp; Point(p3) = {CX+s,CY,0,ls2};
    p4 = newp; Point(p4) = {CX,CY-s,0,ls2};
    
    l1 = newl; Circle(l1) = {p1,p,p2};
    l2 = newl; Circle(l2) = {p2,p,p3};
    l3 = newl; Circle(l3) = {p3,p,p4};
    l4 = newl; Circle(l4) = {p4,p,p1};

    sp = news; Line Loop(sp) = {l1,l2,l3,l4};
    Plane Surface(sp) = {sp};
    loops[] += {sp};

Return

CX = 615; CY = 285; S=10;
Call circle;
CX = 450; CY = 325; S=25;
Call circle;
CX = 525; CY = 325; S=25;
Call circle;
CX = 625; CY = 325; S=25;
Call circle;
CX = 350; CY = 375; S=25;
Call circle;
CX = 450; CY = 400; S=50;
Call circle;
CX = 550; CY = 400; S=50;
Call circle;
CX = 675; CY = 400; S=25;
Call circle;
CX = 350; CY = 475; S=25;
Call circle;
CX = 525; CY = 475; S=25;
Call circle;
CX = 450; CY = 500; S=50;
Call circle;
CX = 600; CY = 500; S=50;
Call circle;
CX = 675; CY = 500; S=25;
Call circle;
CX = 525; CY = 525; S=25;
Call circle;
CX = 425; CY = 575; S=25;
Call circle;
CX = 475; CY = 575; S=25;
Call circle;
CX = 550; CY = 600; S=50;
Call circle;
CX = 625; CY = 600; S=25;
Call circle;
CX = 350; CY = 625; S=25;
Call circle;
CX = 425; CY = 650; S=25;
Call circle;
CX = 635; CY = 665; S=10;
Call circle;
CX = 500; CY = 675; S=25;
Call circle;
CX = 575; CY = 675; S=25;
Call circle;
CX = 335; CY = 715; S=10;
Call circle;

sp=news;
Plane Surface(sp) = {1, loops[]};

Physical Surface(1) = {sp};
Physical Surface(2) = {loops[]};
Physical Line(1) = {1};
Physical Line(2) = {3};
Physical Line(3) = {2};
Physical Line(4) = {4};