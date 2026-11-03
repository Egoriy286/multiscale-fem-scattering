#include <dolfin.h>
#include <cmath>
#include "Parabolic.h"
    
// kHat*u+coords
using namespace dolfin;

class Koeff_k2 : public Expression{
    double** arr;
    double hx, hy;
    double _scale;

public:
    Koeff_k2(double scale):_scale(scale){
    };
    
    void load(std::string filename, int N1, int N2) {
        double Lx = 60;//100; 
        double Ly = 60;//20;
        hx = Lx/(N1-1);
        hy = Ly/(N2-1);
        std::ifstream in(filename.data());
        arr = new double*[N1];
        for (int i = 0; i < N1; i++) {
            arr[i] = new double[N2];
        }
        for (int j = 0; j < N2; j++) {
            for (int i = 0; i < N1; i++) {
                in >> arr[i][j];
            }
        }
        in.close();
        info("loaded");
    }
    
    void eval(Array<double>& values, const Array<double>& x) const{
        int i = x[0]/hx;
        int j = x[1]/hy;
        values[0] = arr[i][j]*_scale;
    }
};

class Koeff_k : public Expression{
    double** arr;
    double hx, hy;
    double _umin, _umax;

public:
    Koeff_k(double umin, double umax):_umin(umin), _umax(umax){
    };
    
    void load(std::string filename, int N1, int N2) {
        double Lx = 60.0; 
        double Ly = 60.0;
        hx = Lx/(N1-1);
        hy = Ly/(N2-1);
        std::ifstream in(filename.data());
        arr = new double*[N1];
        for (int i = 0; i < N1; i++) {
            arr[i] = new double[N2];
        }
        
        for (int j = 0; j < N2; j++) {
            for (int i = 0; i < N1; i++) {
                in >> arr[i][j];
            }
        }
        for (int i = 0; i < N1; i++) {
            for (int j = 0; j < N2; j++) {
                if(arr[i][j] == 2)
                    arr[i][j] = _umin;//1.0e9;
                else
                    arr[i][j] = _umax;//18.0e9;
            }
        }
        in.close();
        info("loaded");
    }
    
    void eval(Array<double>& values, const Array<double>& x) const{
        int i = x[0]/hx;
        int j = x[1]/hy;
        values[0] = arr[i][j];
    }
};


class Permeability: public Expression{
private:
    double kmax, kmin;
    double a;
    double _mid;
public:
    Permeability(double mid = 0.1){
        kmax = 1.0;
        kmin = 0.0;
        a = 2;//2;
        _mid = mid;
    }
    void eval(Array<double>& values, const Array<double>& x) const{
        double v = dolfin::rand();// random number from 0 to 1
        
        double rr = ( exp(v*a)-1)/( exp(a)-1 );// random number from 0 to 1 -> exp
        double vv = kmin + rr*kmax;
        values[0] = (vv < _mid)?kmin:kmax;
    } 
    
};



class DomainMarker : public SubDomain{
private:
//    Function* _koef;
    Expression* _koef;
    double kmax, meps;
public:
//    DomainMarker(FunctionSpace& V){
//        kmax = 1.0; meps = 1.0e-5;
//        // load
//        Mesh permMesh("./perm/mesh.xml");
//        Parabolic::CoefficientSpace_w0 permV(permMesh);
//        Function perm(permV, "./perm/k.xml");
//        _koef = new Function(V);
//        _koef->interpolate(perm);
//    }
    
    DomainMarker(Expression& perm){
        kmax = 1.0; meps = 1.0e-5;
        _koef = &perm;
    }

    bool inside(const Array<double>& x, bool on_boundary) const{
       double val = _koef->operator ()(x[0], x[1]);
       if(std::abs(val-kmax) < meps)
            return true;
        return false;
    }
    
};

class NonlinKoeffBase : public Expression{
protected:
    Function *_f;
public:
    void setF(Function &f){
        _f = &f; 
    }
    
    virtual double getValue(double c) const = 0;
    
    virtual void eval(Array<double>& values, const Array<double>& x) const{
        double val = _f->operator ()(x[0], x[1]);
        values[0] = this->getValue(val);
    }
};

class NonlinKoeff_ak : public NonlinKoeffBase{
private:
    double _phi, _Cms, _kk;
public:
    NonlinKoeff_ak(double phi, double Cms, double kk){
        _phi  = phi;
        _Cms = Cms;
        _kk  = kk;
    };
    
    virtual double getValue(double c) const{
//        double dFdc = _Cms * _kk/pow(1.0 + _kk * c, 2.0);
        double dFdc = -2*_Cms * _kk * _kk/pow(1.0 + _kk * c, 3.0);
        double val = _phi + (1.0 - _phi) * dFdc;
        return val;
    }
};

class NonlinKoeff_bk : public NonlinKoeffBase{
private:
    double _phi, _Cms, _kk;
    double _Ds, _Dk;
public:
    NonlinKoeff_bk(double phi, double Cms, double kk, double Ds, double Dk){
        _phi  = phi;
        _Cms = Cms;
        _kk  = kk;
        
        _Ds = Ds;
        _Dk = Dk;
    };
    
    virtual double getValue(double c) const{
//        double dFdc = _Cms * _kk/pow(1.0 + _kk * c, 2.0);
        double dFdc = -2*_Cms * _kk * _kk/pow(1.0 + _kk * c, 3.0);
        double val = _phi*_Dk + (1.0 - _phi)*_Ds*dFdc;
        return val;
    }
};

class NonlinKoeff : public NonlinKoeffBase{
private:
    double _a1, _a2;
public:
    NonlinKoeff(double a1, double a2){
        _a1 = a1;
        _a2 = a2;
    };
    
    virtual double getValue(double c) const{
        double val = _a1 + c*_a2;
        return val;
    }
};

class NonlinKoeff2 : public NonlinKoeffBase{
private:
    double _a1, _a2;
    Expression* _a3;
    double _Cms, _kk;
public:
    NonlinKoeff2(double a1, double a2, Expression& a3, double Cms, double kk){
        _a1 = a1; _a2 = a2; _a3 = &a3;
        _Cms = Cms; 
        _kk  = kk;
    };
    
    virtual void eval(Array<double>& values, const Array<double>& x) const{
//        double c = ccc;//_f->operator ()(x[0], x[1]);
        double c = _f->operator ()(x[0], x[1]);
        double val3 = _a3->operator ()(x[0], x[1]);
        values[0] = this->getValue(c, val3);
    }
    
    virtual double getValue(double c) const{
        return 0;
    }
    
    double ccc;
    void setConst(double c){
        ccc = c;
    }
    
    virtual double getValue(double c, double a3) const{
        double dFdc = _Cms * _kk/pow(1.0 + _kk * c, 2.0);
//        double dFdc = -2*_Cms * _kk * _kk/pow(1.0 + _kk * c, 3.0);
        double val = _a1 + _a2*dFdc + a3*c;
        return val;
    }
};


void saveKoefficient(std::string fileName, Mesh& mesh, Expression& v){
    Parabolic::Form_a::TrialSpace VF(mesh);
    Function funcK(VF);
    funcK = v;
    
    File fileR(fileName);
    fileR << funcK;
}



