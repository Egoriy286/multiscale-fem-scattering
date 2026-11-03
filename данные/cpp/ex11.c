
static char help[] = "Solves a linear system in parallel with KSP.\n\n";
#include <petscksp.h>
#include <time.h>


void saveVector(bool asRealPart, int dim, PetscScalar *array2, std::string uname){
    PetscReal *array;
    PetscMalloc1(dim, &array);
    for(int j = 0; j < dim; j++){
        if(asRealPart)
            array[j] = (double)PetscRealPart(array2[j]);
        else
            array[j] = (double)PetscImaginaryPart(array2[j]);
    }
    
    int fd;
    PetscViewer viewerRhs2;
    PetscViewerBinaryOpen(PETSC_COMM_SELF, uname.data(), FILE_MODE_WRITE, &viewerRhs2);
    PetscViewerBinaryGetDescriptor(viewerRhs2, &fd);
    PetscBinaryWrite(fd, array, dim, PETSC_DOUBLE, PETSC_FALSE);//PETSC_SCALAR, PETSC_FALSE);
    //        VecView(Pb1, viewerRhs2);
    printf("save %s %d \n", uname.data(), dim);
    PetscViewerDestroy(&viewerRhs2);
    
    PetscFree(array);
}

Vec loadVector(bool asRealPart, int dim, std::string rhsname){
    PetscReal *avec2;
    PetscScalar *avec;
    PetscMalloc1(dim, &avec);
    PetscMalloc1(dim, &avec2);
    
    Vec b2;
    VecCreateSeq(PETSC_COMM_SELF, dim, &b2);
    
    int fd;
    PetscViewer viewerRhs1;
    PetscViewerBinaryOpen(PETSC_COMM_SELF, rhsname.data(), FILE_MODE_READ, &viewerRhs1);
    PetscViewerBinaryGetDescriptor(viewerRhs1, &fd);
    PetscBinaryRead(fd, avec2, dim, PETSC_DOUBLE);//PETSC_SCALAR);
    for(int j = 0; j < dim; j++){
        if(asRealPart)
            avec[j] = avec2[j];
        else
            avec[j] = avec2[j]*PETSC_i;
        VecSetValues(b2, 1, &j, &avec[j], INSERT_VALUES);
        
        if(std::abs(avec2[j]) > 1.0e-10)
            printf("%d: v %g \n", j, avec2[j]);
    }
//    VecLoad(b2, viewerRhs1);
    VecAssemblyBegin(b2);
    VecAssemblyEnd(b2);
    PetscViewerDestroy(&viewerRhs1);
    
    PetscInt m;
    VecGetSize(b2, &m);
    printf("Vector size %d \n", m);
    
//    PetscFree(avec);
    PetscFree(avec2);
    
    return b2;
}

Mat loadMatrix(bool asRealPart, int dim, std::string fname){
    std::string nameS1 = fname + "1.dat";
    std::string nameS2 = fname + "2.dat";
    std::string nameSo = fname + "o.dat";
    printf("START: Matrix load\n");
    int fd, fd1, fd2;
    
    PetscInt *olen;
    PetscViewer viewer1o;
    PetscViewerBinaryOpen(PETSC_COMM_SELF, nameSo.data(), FILE_MODE_READ, &viewer1o);
    PetscViewerBinaryGetDescriptor(viewer1o, &fd);
    PetscMalloc1(dim, &olen);
    PetscBinaryRead(fd, olen, dim, PETSC_INT);
    PetscViewerDestroy(&viewer1o);
    
    
    Mat A;
    int maxnz = 0;
    for(int j = 0; j < dim; j++){
        maxnz = (maxnz < olen[j])?olen[j]:maxnz;
    //printf("olen[%d] = %d\n", j, olen[j]);
    }
    MatCreateSeqAIJ(PETSC_COMM_SELF, dim, dim, maxnz, 0, &A);
    printf("Matrix nz %d\n", maxnz);
    
    
    PetscInt *cols;
    PetscReal *vals;
    PetscScalar *vals2;
    PetscMalloc1(maxnz, &cols);
    PetscMalloc1(maxnz, &vals);
    PetscMalloc1(maxnz, &vals2);
    
    PetscViewer viewer11, viewer12;
    PetscViewerBinaryOpen(PETSC_COMM_SELF, nameS1.data(), FILE_MODE_READ, &viewer11);
    PetscViewerBinaryOpen(PETSC_COMM_SELF, nameS2.data(), FILE_MODE_READ, &viewer12);
    PetscViewerBinaryGetDescriptor(viewer11, &fd1);
    PetscViewerBinaryGetDescriptor(viewer12, &fd2);
    for(int row = 0; row < dim; row++){ // rows
        PetscBinaryRead(fd1, vals, olen[row], PETSC_DOUBLE);//PETSC_SCALAR);
        PetscBinaryRead(fd2, cols, olen[row], PETSC_INT);
//        printf("--- %d with nz %d ---\n", row, olen[row]);
        for(int j = 0; j < olen[row]; j++){
            if(asRealPart)
                vals2[j] = vals[j];
            else
                vals2[j] = PETSC_i*vals[j];
//            printf("c %d, v %g \n", cols[j], vals[j]);
//            if(std::abs(vals[j]) > 1.0e-10)
//                printf("%d with nz %d: c %d, v %g \n", row, olen[row], cols[j], vals[j]);
        }
        MatSetValues(A, 1, &row, olen[row], cols, vals2, INSERT_VALUES);
    }
    //    MatLoad(A, viewerS);
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
    PetscViewerDestroy(&viewer11);
    PetscViewerDestroy(&viewer12);
    
    PetscInt dim1, dim2;
    MatGetSize(A, &dim1, &dim2);
    printf("END: Matrix size [%d, %d]\n", dim1, dim2);
    
    PetscFree(olen);
    PetscFree(cols);
    PetscFree(vals);
    PetscFree(vals2);
    
    return A;
}

int main(int argc,char **args){
    PetscInitialize(&argc,&args,(char*)0,help);
    
    /*
     * SIZE
     */
//    int dim = 6561;
    int dim = atoi(args[1]);
    
    // output
    std::string outname = "resFiles/";
    std::string unameR = outname + "uR.dat";
    std::string unameI = outname + "uI.dat";
    // input
    std::string dirname = "binFiles/";
    std::string rhsname1 = dirname + "bRe.dat";
    std::string rhsname2 = dirname + "bIm.dat";
    std::string nameA1 = dirname + "ARe";
    std::string nameA2 = dirname + "AIm";

    
    /*
     * LOAD MATRIX
     */
    Mat A = loadMatrix(true,  dim, nameA1);// Re
    printf("1\n");
    Mat AIm = loadMatrix(false, dim, nameA2);// Im
    printf("2\n");
    
    /*
     * LOAD VECTOR
     */
    Vec b = loadVector(true,  dim, rhsname1);// Re
    printf("3\n");
    Vec bIm = loadVector(false, dim, rhsname2);// Im
    printf("4\n");

//    printf("----\n");
//    PetscScalar val;
//    for(int i = 0; i < dim; i++){
//        VecGetValues(b, 1, &i, &val);
//        double vr = (double)PetscRealPart(val);
//        double vi = (double)PetscImaginaryPart(val);
//        if(std::abs(vr) > 1.0e-10)
//            printf("%d: v %g \n", i, vr);
//    }
    
    /*
     * ADD MATRICES
     */
    MatAXPY(A, 1.0, AIm, DIFFERENT_NONZERO_PATTERN);
    VecAXPY(b, 1.0, bIm);

    /*
     * SOLVE COMPLEX
     */
    Vec x;
    VecCreateSeq(PETSC_COMM_SELF, dim, &x);
    
    KSP ksp;
    PetscInt its;
    KSPCreate(PETSC_COMM_SELF, &ksp);
    KSPSetOperators(ksp, A, A);
    KSPSetFromOptions(ksp);

    time_t start,end;
    time (&start);
    printf ("SOLVE");
    
    KSPSolve(ksp, b, x);
    
    time (&end);
    double dif = difftime (end,start);
    printf ("SOLVE time is %lf seconds.", dif);
    
    KSPGetIterationNumber(ksp, &its);
    
//    PetscScalar *arrs;
//    VecGetArray(x, &arrs);
//    for(int j = 0; j < dim; j++){
//        if(std::abs(arrs[j]) > 1.0e-10)
//            printf("%d: v %g + %g i\n", j, (double)PetscRealPart(arrs[j]),(double)PetscImaginaryPart(arrs[j]));
//    }
//    
//    printf("Iter = %d\n", its);
    
    PetscScalar *array2;
    VecGetArray(x, &array2);
    saveVector(true, dim, array2, unameR);
    saveVector(false, dim, array2, unameI);
    
//    VecView(x, PETSC_VIEWER_STDOUT_WORLD);
    
    KSPDestroy(&ksp);
    VecDestroy(&x);
    VecDestroy(&b);
    VecDestroy(&bIm);
    MatDestroy(&A);
    MatDestroy(&AIm);
    PetscFinalize();
    return 0;
}
