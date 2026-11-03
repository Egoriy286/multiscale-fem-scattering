#ifndef ELLIPTICPROBLEM_H
#define ELLIPTICPROBLEM_H

#include "CElliptic.h"
#include "MsProblemComplex.h"
#include <fstream>

using namespace dolfin;


class SDMarkers : public Expression{
    double _v1, _v2, _v3, _v4;
    MeshFunction<size_t> *_subdomainsFine;

public:
    SDMarkers(MeshFunction<size_t> &subdomainsFine, double v1, double v2, double v3, double v4){
        _subdomainsFine = &subdomainsFine;
        _v1 = v1;
        _v2 = v2;
        _v3 = v3;
        _v4 = v4;
    };
    
    void eval(Array<double>& values, const Array<double>& x, const ufc::cell& cell) const{
        if(_subdomainsFine->operator [](cell.index) == 1)
            values[0] = _v1;
        else if(_subdomainsFine->operator [](cell.index) == 2)
            values[0] = _v2;
        else if(_subdomainsFine->operator [](cell.index) == 3)
            values[0] = _v3;
        else
            values[0] = _v4;
    }
};


class CEllipticProblem: public MsProblemComplex{
public:

    CEllipticProblem(CoarseWrapper& coarseWrapper, 
            Expression &Ax, Expression &n, Expression &k, Expression &fRe, Expression &fIm)
    :MsProblemComplex(coarseWrapper){
        _fRe = &fRe;
        _fIm = &fIm;
        _Ax = &Ax;
        _k = &k;
        _n = &n;
        _zero = new Constant(0.0);
        
        _V  = new CElliptic::Form_a::TrialSpace(_meshWrapper->getFineMesh());
        _u  = new Function(*_V);
        // generate forms
        CElliptic::Form_a* a = new CElliptic::Form_a(*_V, *_V);
        CElliptic::Form_b* b = new CElliptic::Form_b(*_V, *_V);
        CElliptic::Form_Lc* L1 = new CElliptic::Form_Lc(*_V);
        CElliptic::Form_Lc* L2 = new CElliptic::Form_Lc(*_V);
        a->Ax = *_Ax;
        a->k = *_k;
        a->n = *_n;
        b->k = *_k;
//        L1->f  = *_fRe;
//        L2->f  = *_fIm;
        L1->Ax = *_Ax;
        L1->k = *_k;
        L1->n = *_n; 
        L1->ui  = *_fRe;
        L2->Ax = *_Ax;
        L2->k = *_k;
        L2->n = *_n;
        L2->ui  = *_fIm;
        _are = a;
        _aim = b;
        _Lre = L1;
        _Lim = L2;
        
        _ARe = new PETScMatrix();
        _AIm = new PETScMatrix();
        _bRe = new PETScVector(MPI_COMM_SELF, _u->vector().get()->size()); 
        _bIm = new PETScVector(MPI_COMM_SELF, _u->vector().get()->size()); 
        
        _havePointSource = false;
        info("init");
    };
    
    void setPointSource(Point pp){
        _havePointSource = true;
        _pp = &pp;
    }
    
    std::string calculateErrors(Function& uf, Function& ums, int Bc, int ind){
        Mesh& mesh = _meshWrapper->getFineMesh();
        // L2 error
        CElliptic::Form_SL2 sumFormL2(mesh);
        sumFormL2.w1 = uf;
        CElliptic::Form_EL2 absFormL2(mesh);
        absFormL2.w1 = uf;
        absFormL2.w2 = ums;
        double sumErrL2 = assemble(sumFormL2);
        double absErrL2 = assemble(absFormL2);
        double errL2 = sqrt(absErrL2)/sqrt(sumErrL2);
        // H1 error
        CElliptic::Form_SH1 sumFormH1(mesh);
        sumFormH1.w1 = uf;
        CElliptic::Form_EH1 absFormH1(mesh);
        absFormH1.w1 = uf;
        absFormH1.w2 = ums; 
        double sumErrH1 = assemble(sumFormH1);
        double absErrH1 = assemble(absFormH1);
        double errH1 = sqrt(absErrH1)/sqrt(sumErrH1);
        int BasisCount = Bc;

        info(" relative errors: Basiscount = %d, L2 = %g, H1 = %g", BasisCount, errL2*100, errH1*100);
        std::ostringstream strOut;        
        strOut << " " << (errL2*100)  << " " << (errH1*100);
        //return strOut.str();
        if(ind==1){
            std::ofstream output1("errors.txt", std::ios_base::app);
            output1<<" --- Re(us) --- \n";
            output1<<"BasisCount = "<<BasisCount<<",\tL2 = "<< 100*errL2 <<",\tH1 = " << 100*errH1 <<"\n";
            output1.close();
        }
        if(ind==2){
            std::ofstream output1("errors.txt", std::ios_base::app);
            output1<<" --- Im(us) --- \n";
            output1<<"BasisCount = "<<BasisCount<<",\tL2 = "<< 100*errL2 <<",\tH1 = " << 100*errH1 <<"\n";
            output1.close();
        }
        if(ind==3){
            std::ofstream output1("errors.txt", std::ios_base::app);
            output1<<" --- Abs(u) --- \n";
            output1<<"BasisCount = "<<BasisCount<<",\tL2 = "<< 100*errL2 <<",\tH1 = " << 100*errH1 <<"\n";
            output1.close();
        }
        return strOut.str();
    }
    
private:
    Expression* _fRe;
    Expression* _fIm;
    Expression* _zero;
    Expression* _Ax;
    Expression* _k;
    Expression* _n;
    
    Point* _pp;
    PointSource* _pf;
    
    virtual void modifyA(){
        // DO NOTHING
    }
    
    bool _havePointSource;
    virtual void modifyb(){
        if(_havePointSource){
            PointSource fwell(*_V, *_pp, 1.0);
            fwell.apply(*_bRe);
        }
    }
    
};

#endif
