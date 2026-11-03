#include "CEllipticProblem.h"
#include <dolfin.h>
#include <complex>

using namespace dolfin;

class MySource : public Expression{
    MeshFunction<size_t> *_subdomainsFine;
    Mesh *_mesh;
    int _type;
    double _k, _vv;
    Point *_d;

public:
    MySource(Mesh& mesh, MeshFunction<size_t> &subdomainsFine, int type, double k, double vv, Point d){
        _mesh = &mesh;
        _subdomainsFine = &subdomainsFine;
        _type = type;
        _k = k;
        _vv = vv;
        _d = &d;
    };
    
    void eval(Array<double>& values, const Array<double>& x, const ufc::cell& cell) const{
//        Cell cell1(*_mesh, cell.index);
//        Point n = cell1.normal(cell.local_facet);
        double kxd = _k*(_d->x()*x[0] +_d->y()*x[1]);
//        double kxd = _k*x[0];
        if(_subdomainsFine->operator [](cell.index) == 1){
            values[0] = 0.0;
        }else{
            values[0] = (_type == 1)?(_vv*cos(kxd)):(_vv*sin(kxd));
        }
    }
};

class MyIncidentU : public Expression{
    MeshFunction<size_t> *_subdomainsFine;
    Mesh *_mesh;
    int _type;
    double _k;
    Point *_d;

public:
    MyIncidentU(Mesh& mesh, MeshFunction<size_t> &subdomainsFine, int type, double k, Point d){
        _mesh = &mesh;
        _subdomainsFine = &subdomainsFine;
        _type = type;
        _k = k;
        _d = &d;
    };
    
    void eval(Array<double>& values, const Array<double>& x, const ufc::cell& cell) const{
//        Cell cell1(*_mesh, cell.index);
//        Point n = cell1.normal(cell.local_facet);
        double kxd = _k*(_d->x()*x[0] +_d->y()*x[1]);
//        double kxd = _k*x[0];
        values[0] = (_type == 1)?cos(kxd):sin(kxd);
    }
};


// export PETSC_DIR="/Users/vmasha/work/projects/ms/petsc/petsc/"
// make ex11
// ./ex11 NNN  -pc_type lu -ksp_type preonly -pc_factor_mat_solver_package mumps
// ./problem 50 ../../data/meshForPHelm-sb1/mesh2 441 ../Ms-Domain/out-h/ 8 1.0 11.0 0.2
// ./problem 50 ../../data/meshForPHelm-sb1/mesh2 121 ../Ms-Domain/out-h/ 8 1.0 11.0 0.2
// ./main 20 ../mesh/mesh2 121 ../spectral/ 10 1.0 11.0 0.2
// Perforated scattering problem in R^2
int main(int argc, char** argv){
    PetscInitialize(&argc, &argv, (char*)0, "");
    parameters["reorder_dofs_serial"] = false;
    parameters["allow_extrapolation"] = true;
    Timer timer("SOLUTION");
    timer.start();
    
    int type = atoi(argv[1]);
    std::string meshDir = "../mesh/mesh2";//"mesh1/mesh"
    int locDom = 121;
    std::string bdir  = "../spectral/";
    int bCount = atoi(argv[2]);
    
    double kval = 2.0;//1.0
    double alpha = 11.0;//11.0
    double Aval = 0.2;//0.2

    std::string resultsDir = "./results/";
    
    // mesh
    Mesh meshFine(meshDir + ".xml");
    MeshFunction<size_t> boundariesFine(meshFine, meshDir + "_facet_region.xml");
    MeshFunction<size_t> subdomainsFine(meshFine, meshDir + "_physical_region.xml");
    File filesubdoms(resultsDir+"sd.pvd");
    filesubdoms << subdomainsFine;
    // mesh wrapper
    CoarseWrapper meshWrapper(meshFine, boundariesFine, subdomainsFine, boundariesFine);
    meshWrapper.setNumberOfLocalDomains(locDom);

    // Coefficient
    Point d(1.0, 0.0);
    Constant k(kval);
    SDMarkers n(subdomainsFine, 1.0, alpha-1, alpha+(alpha-1.0)/2, alpha+(alpha-1.0)/4);
    SDMarkers Ax(subdomainsFine, 1.0, Aval, Aval*2, Aval*4);
    //SDMarkers n(subdomainsFine, 1.0, alpha-1, alpha+(alpha-1.0)/2, alpha+(alpha-1.0)/4);
    //SDMarkers Ax(subdomainsFine, 1.0, Aval, Aval*2, Aval*4);
    //    MySource fRe(meshFine, subdomainsFine, 1, kval, kval*kval*(alpha - 1.0), d);
    //    MySource fIm(meshFine, subdomainsFine, 2, kval, kval*kval*(alpha - 1.0), d);
    MyIncidentU uiRe2(meshFine, subdomainsFine, 1, kval, d);
    MyIncidentU uiIm2(meshFine, subdomainsFine, 2, kval, d);
    
    //    Constant n(1.0);
    //    SDMarkers k(subdomainsFine, 10.0, 20.0);
    //    SDMarkers k(subdomainsFine, 10.0, 10.0);
    //    Constant fRe(0.0);
    //    Constant fIm(0.0);
    //    Constant zero(0.0);
    
    CEllipticProblem problem(meshWrapper, Ax, n, k, uiRe2, uiIm2);
    //    problem.setPointSource(Point(0.5, 0.5));
    //    problem.addDirichletBC(zero, 1);
    //    problem.addDirichletBC(zero, 2);
    //    problem.addDirichletBC(zero, 3);
    //    problem.addDirichletBC(zero, 4);
    
    Function inRe(problem.V());
    inRe = uiRe2;
    Function inIm(problem.V());
    inIm = uiIm2;
    
    if(type == 50){
        info(" ----- fine solution: generate A and b ----- ");
        problem.generateSystem();
        info(" ----- fine solution: generate A and b ----- ");
    }
    if(type == 51){
        int NN = problem.u().vector().get()->size();
        info(" ----- fine solution: uRe ----- ");
        problem.saveToPvd("resFiles/uR.dat", resultsDir+"uR.pvd");
        Function scRe(problem.V());
        for(int i = 0; i < NN; i++){
            double us = problem.u().vector().get()->getitem(i);
            scRe.vector().get()->setitem(i, us);
        }
        scRe.vector().get()->apply("insert");
        
        info(" ----- fine solution: uIm ----- ");
        problem.saveToPvd("resFiles/uI.dat", resultsDir+"uI.pvd");
        Function scIm(problem.V());
        for(int i = 0; i < NN; i++){
            double us = problem.u().vector().get()->getitem(i);
            scIm.vector().get()->setitem(i, us);
        }
        scIm.vector().get()->apply("insert");
        
        info(" ----- fine solution: uAbs ----- ");
        Function uAbs(problem.V());
        Function scAbs(problem.V());
        Function inAbs(problem.V());
        for(int i = 0; i < NN; i++){
            double inReVal = inRe.vector().get()->getitem(i);
            double inImVal = inIm.vector().get()->getitem(i);
            inAbs.vector().get()->setitem(i, std::sqrt(inReVal*inReVal + inImVal*inImVal));
            double scReVal = scRe.vector().get()->getitem(i);
            double scImVal = scIm.vector().get()->getitem(i);
            scAbs.vector().get()->setitem(i, std::sqrt(scReVal*scReVal + scImVal*scImVal));
            double uReVal = inReVal + scReVal;
            double uImVal = inImVal + scImVal;
            uAbs.vector().get()->setitem(i, std::sqrt(uReVal*uReVal + uImVal*uImVal));
        }
        uAbs.vector().get()->apply("insert");
        scAbs.vector().get()->apply("insert");
        inAbs.vector().get()->apply("insert");
        File ffAbs1(resultsDir+"uAbs.pvd");
        ffAbs1 << uAbs;
        File ffAbs2(resultsDir+"scAbs.pvd");
        ffAbs2 << scAbs;
        
        File ffAbs31(resultsDir+"inURe.pvd");
        ffAbs31 << inRe;
        File ffAbs32(resultsDir+"inUIm.pvd");
        ffAbs32 << inIm;
        
        File fxml_uRe(resultsDir+"uR.xml");
        fxml_uRe << scRe;
        File fxml_uIm(resultsDir+"uI.xml");
        fxml_uIm << scIm;
        File fxml_uAbs(resultsDir+"uAbs.xml");
        fxml_uAbs << uAbs;
        info(" ----- fine solution: u ----- ");
    }
    
    if(type == 20){
        info(" ----- coarse solution: generate Ac and bc ----- ");
        problem.eCount = bCount;
        problem.loadMsBasis(bdir, bdir + "basis");
        // problem.loadMsBasisByLambda(bdir, bdir + "eigen/");
        problem.generateR();
        problem.generateSystem();
        problem.generateSystemCoarse();
        info(" ----- coarse solution: generate Ac and bc ----- ");
    }
    if(type == 21){
        info(" ----- coarse solution: Ru ----- ");
        problem.eCount = bCount;
        problem.loadMsBasis(bdir, bdir + "basis");
        problem.generateR();
        
        int NN = problem.u().vector().get()->size();
        info(" ----- coarse solution: uRe ----- ");
        problem.saveToPvdCoarse("resFiles/uR.dat", resultsDir+"uR2.pvd");
        Function scRe(problem.V());
        for(int i = 0; i < NN; i++){
            double us = problem.u().vector().get()->getitem(i);
            scRe.vector().get()->setitem(i, us);
        }
        scRe.vector().get()->apply("insert");
        
        info(" ----- coarse solution: uIm ----- ");
        problem.saveToPvdCoarse("resFiles/uI.dat", resultsDir+"uI2.pvd");
        Function scIm(problem.V());
        for(int i = 0; i < NN; i++){
            double us = problem.u().vector().get()->getitem(i);
            scIm.vector().get()->setitem(i, us);
        }
        scIm.vector().get()->apply("insert");

        info(" ----- coarse solution: uAbs ----- ");
        Function uAbs(problem.V());
        Function scAbs(problem.V());
        Function inAbs(problem.V());
        for(int i = 0; i < NN; i++){
            double inReVal = inRe.vector().get()->getitem(i);
            double inImVal = inIm.vector().get()->getitem(i);
            inAbs.vector().get()->setitem(i, std::sqrt(inReVal*inReVal + inImVal*inImVal));
            double scReVal = scRe.vector().get()->getitem(i);
            double scImVal = scIm.vector().get()->getitem(i);
            scAbs.vector().get()->setitem(i, std::sqrt(scReVal*scReVal + scImVal*scImVal));
            double uReVal = inReVal + scReVal;
            double uImVal = inImVal + scImVal;
            uAbs.vector().get()->setitem(i, std::sqrt(uReVal*uReVal + uImVal*uImVal));
        }
        uAbs.vector().get()->apply("insert");
        scAbs.vector().get()->apply("insert");
        inAbs.vector().get()->apply("insert");
        File ffAbs1(resultsDir+"uAbs2.pvd");
        ffAbs1 << uAbs;
        // load fine-scale solution
        Function fscRe(problem.V());
        Function fscIm(problem.V());
        Function fuAbs(problem.V());
        File fxml_uRe(resultsDir+"uR.xml");
        fxml_uRe >> fscRe;
        File fxml_uIm(resultsDir+"uI.xml");
        fxml_uIm >> fscIm;
        File fxml_uAbs(resultsDir+"uAbs.xml");
        fxml_uAbs >> fuAbs;
        // error
        info(" --- Re(us) --- ");
        problem.calculateErrors(fscRe, scRe, bCount, 1);
        info(" --- Im(us) --- ");
        problem.calculateErrors(fscIm, scIm, bCount, 2);
        info(" --- Abs(u) --- ");
        problem.calculateErrors(fuAbs, uAbs, bCount, 3);
        info(" ----- coarse solution: Ru ----- ");
        info("DONE");
    }
    
    PetscFinalize();
    return 0;
}
