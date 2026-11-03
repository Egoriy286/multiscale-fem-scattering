#ifndef MSBLOCK_H
#define MSBLOCK_H

#include <math.h>
#include <dolfin.h>
#include <iostream>
#include <fstream> // std::ifstream

using namespace dolfin;

class MsBlock {
public:
    MsBlock(int ind, std::string dir, int eCount){
        _eCount = eCount;
        _allBnBasis = false;
        
        _dir = dir;
        _ind = ind;
        
         globalInds.clear();
    }
    virtual ~MsBlock(){
        MatDestroy(&R); 
        MatDestroy(&RT);
    }
    
    void setBoundaryBasisPercentForLoad(double percentBn){
        _allBnBasis = true;
        _percentBn = percentBn;
    }
    
     virtual void loadDOF(){
        std::ostringstream fileNameDOF;
        fileNameDOF << _dir << "DofMap/" << _ind << ".txt";
        std::ifstream doffile(fileNameDOF.str().data());
        info("Putin DOF = %s", fileNameDOF.str().data());
        int bi,gi, lN;
        doffile >> lN;
        for(int li = 0; li < lN; li++){
            doffile >> gi;
            globalInds.push_back(gi);
        }
        doffile.close();
        info("\nload for %d is %d", _ind, lN);
    }
    
    virtual void loadMSBasis(std::string cdir){
        // load count for boundary basis
        std::ostringstream fileNameCount;
        fileNameCount << cdir << "/ecount-" << _ind << ".txt";
        std::ifstream fileCountEj(fileNameCount.str().data());
        int bnBasisOnEdgeCountEj = 0;
        fileCountEj >> bnBasisOnEdgeCountEj;
        fileCountEj >> bnBasisOnEdgeCountEj;
        fileCountEj.close();
        //info("-->>-->> %d", bnBasisOnEdgeCountEj);
    
        int curSize = 0;
        if(_allBnBasis)// change number of basis functions
                curSize = _percentBn * bnBasisOnEdgeCountEj;

        if(_eCount > bnBasisOnEdgeCountEj)
            curSize = bnBasisOnEdgeCountEj;
        else
            curSize = _eCount;
        // info("-->> Change R%d from %d to %d", _ind, _eCount, curSize);

        int Nw = globalInds.size();
        int Nb = curSize;
        info("w%d: %d, %d", _ind, Nw, Nb);
        MatCreateSeqAIJ(PETSC_COMM_SELF, Nb, Nw, Nw, 0, &R);

        for(int ek = 0; ek < curSize; ek++){
            std::ostringstream fileNameBn;
            fileNameBn << cdir << "/bcount" << ek << "omega" << _ind << "-z";
            std::ifstream bfile(fileNameBn.str().data());
            int bbN1, bbN2;
            bfile >> bbN1 >> bbN2; // get size
            info("bb:%s %d, %d", fileNameBn.str().data(), bbN1, bbN2);
            double val;
            for(int li = 0; li < bbN2; li++){
                bfile >> val;
                MatSetValues(R, 1, &ek, 1, &li ,&val, INSERT_VALUES);
            }
            bfile.close();
        }
        
        MatAssemblyBegin(R, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(R, MAT_FINAL_ASSEMBLY);
        // RT
        MatTranspose(R, MAT_INITIAL_MATRIX, &RT); 
        info("R%d [%d, %d]", _ind, getCoarseSize(), getFineSize());
    }
    
    virtual void loadMSBasisDG(std::string cdir){
        std::ostringstream fileNameCountEj;
        fileNameCountEj << cdir << "/ecount-" << _ind << ".txt";
        std::ifstream fileCountEj(fileNameCountEj.str().data());
        //info("%s", fileNameCountEj.str().data());
        int basisCount = 0;
        fileCountEj >> basisCount;
        fileCountEj.close();
        //info("basisCount = %d", basisCount);

        int curSize = 0;
        if(_allBnBasis)// change number of basis functions
             curSize = _percentBn * basisCount;
        if(_eCount > basisCount)
            curSize = basisCount;
        else
            curSize = _eCount;
        
        int Nw = globalInds.size();
        int Nb = curSize;
        //info("w%d: %d, %d", _ind, Nw, Nb);
        MatCreateSeqAIJ(PETSC_COMM_SELF, Nb, Nw, Nw, 0, &R);
         // ---- boundary basis ----
      
        int eCounter = 0;
        
        for(int ek = 0; ek < Nb; ek++){
            //info("eCounter = %d", eCounter);
            std::ostringstream fileNameBn;
            fileNameBn << cdir << "/bcount" << ek << "omega" << _ind << "-z";
            std::ifstream bfile(fileNameBn.str().data());
            int bbN1, bbN2;
            bfile >> bbN2; // get size
            //info("bb:%s %d, %d", fileNameBn.str().data(), bbN1, bbN2);
            double val;
            for(int li = 0; li < bbN2; li++){
                bfile >> val;
                //info("val[%d] = %lf", li, val);
                MatSetValues(R, 1, &eCounter, 1, &li, &val, INSERT_VALUES);
            }
            eCounter++;
            bfile.close();
        }
       
        MatAssemblyBegin(R, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(R, MAT_FINAL_ASSEMBLY);
        // RT
        MatTranspose(R, MAT_INITIAL_MATRIX, &RT); 
        
        //info("R%d [%d, %d]", _ind, getCoarseSize(), getFineSize());
    }
   
    int getGlobalByLocal(int li){
        return globalInds[li];
    }
    
    double getRItem(int i, int j){
        double val;
        MatGetValues(R, 1, &i, 1, &j, &val);
        return val;
    }
    
    int getCoarseSize(){
        PetscInt Nc, Nf;
        MatGetSize(R, &Nc, &Nf);
        return Nc;
    }
    
    int getFineSize(){
        PetscInt Nc, Nf;
        MatGetSize(R, &Nc, &Nf);
        //info("Nc = %d, Nf = %d", Nc, Nf);
        return Nf;
    }
    
protected:
    std::vector<int> globalInds;

    int Nc, Nf;
    Mat R, RT;
    
    int _eCount;
    bool _allBnBasis;
    double _percentBn;
    
    int _ind;
    std::string _dir;
    
};
#endif	/* MSBLOCK_H */
