#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

//#include "transferGPU.h"

double *u1, *v1, *w1;
double *u2, *v2, *w2;

void readData();
void transfer();
void writeData(
    double *arr_h, const char *fname, const int myid, const int size )
{
    char path[100];
    sprintf( path, "./result/%s_%d.bin", fname, myid );

    FILE *data;
    data = fopen(path, "wb");
    fwrite( arr_h, sizeof(double), size, data );
    fclose( data );
}
void OutputData();


int main(int argc, char *argv[])
{
    size_t nBytes = 128 * 128 * 128 * sizeof(double);

    u1 = (double*)malloc(nBytes);   memset(u1, 0.0, nBytes);
    v1 = (double*)malloc(nBytes);   memset(v1, 0.0, nBytes);
    w1 = (double*)malloc(nBytes);   memset(w1, 0.0, nBytes);

    printf("aslk;jdfasldjflaskdj\n");

    mallocData();
    readData();
    transfer();
    OutputData();
    //transferAllData();

    free(u1);   free(v1);   free(w1);
    free(u2);   free(v2);   free(w2);

    return 0;
}

void mallocData() {
    size_t nBytes = 385 * 961 * 256* sizeof(double);

    u2 = (double*)malloc(nBytes);   memset(u2, 0.0, nBytes);
    v2 = (double*)malloc(nBytes);   memset(v2, 0.0, nBytes);
    w2 = (double*)malloc(nBytes);   memset(w2, 0.0, nBytes);
}

void readData() {
    FILE *data;
    if((data = fopen("128x128x128_Retau=180_Poiseuille_channel.dat", "r")) == NULL) {
        printf("Read data error, exit...\n");
        exit(1);
    }

    double  x, y, z, p;
    int i,j,k;
    int index;
    int NX = 128, NY = 128, NZ = 128;
    for( i = 0; i < NY; i++ ){
    for( k = 0; k < NZ; k++ ){
    for( j = 0; j < NX; j++ ){

        index = j*NX*NZ + k*NX + i;

        fscanf( data, "%lf", &x );
        fscanf( data, "%lf", &y );
        fscanf( data, "%lf", &z );
        fscanf( data, "%lf", &v1[index] );
        fscanf( data, "%lf", &w1[index] );
        fscanf( data, "%lf", &u1[index] );
        fscanf( data, "%lf", &p );

        v1[index] = v1[index] / 4.0;
        w1[index] = w1[index] / 4.0;
        u1[index] = u1[index] / 4.0;

    }}}

    fclose( data );
}

void transfer() {
    int i,j,k;
    int NX1 = 128, NY1 = 128, NZ1 = 128;
    int NX2 = 385, NY2 = 961, NZ2 = 256;

    for( k = 0; k < NZ2;   k++ ){
    for( j = 0; j < NY2-1; j++ ){
    for( i = 0; i < NX2-1; i++ ){

        int index1 = (int)((double)j/NY2*NY1)*NX1*NZ1 + (int)((double)k/NZ2*NZ1)*NX1 + (int)((double)i/NX2*NX1);
        int index2 = j*NX2*NZ2 + k*NX2 + i;

        w2[index2] = w1[index1] * 0.068;			//0.066-> initial = ~0.057 
        v2[index2] = fabs(v1[index1]) * 0.068;
        u2[index2] = u1[index1] * 0.068;

    }}}

    for( j = 0; j < NY2; j++ ){
    for( k = 0; k < NZ2; k++ ){

        w2[j*NX2*NZ2 + k*NX2 + (NX2-1)] = w2[j*NX2*NZ2 + k*NX2 + 0];
        v2[j*NX2*NZ2 + k*NX2 + (NX2-1)] = v2[j*NX2*NZ2 + k*NX2 + 0];
        u2[j*NX2*NZ2 + k*NX2 + (NX2-1)] = u2[j*NX2*NZ2 + k*NX2 + 0];

    }}

    for( i = 0; i < NX2; i++ ){
    for( k = 0; k < NZ2; k++ ){

        w2[(NY2-1)*NX2*NZ2 + k*NX2 + i] = w2[0*NX2*NZ2 + k*NX2 + i];
        v2[(NY2-1)*NX2*NZ2 + k*NX2 + i] = v2[0*NX2*NZ2 + k*NX2 + i];
        u2[(NY2-1)*NX2*NZ2 + k*NX2 + i] = u2[0*NX2*NZ2 + k*NX2 + i];

    }}
}

void OutputData() {
    printf("11111111\n");
    double *u, *v, *w;
    int NP = 8;
    int NX = 385;           int NY = 961;               int NZ = 256;
    int NX6 = NX + 6;       int NYD = (NY-1)/NP+1;      int NZ6 = NZ + 6;
                            int NYD6 = NYD+6;

    size_t nBytes = NX6 * NYD6 * NZ6 * sizeof(double);
    u = (double*)malloc(nBytes);    memset(u, 0.0, nBytes);
    v = (double*)malloc(nBytes);    memset(v, 0.0, nBytes);
    w = (double*)malloc(nBytes);    memset(w, 0.0, nBytes);
    printf("11111111\n");
    int myid;
    for( myid = 0; myid < NP; myid++ ) {
        printf("11111111\n");

        int i,j,k;
        int j_global;
        for( k = 0; k < NZ;  k++ ){
        for( j = 0; j < NYD; j++ ){
        for( i = 0; i < NX;  i++ ){

            j_global = (NYD-1)*myid + j;

            int idx = j_global*NX*NZ + k*NX + i;

            int index = j*NX6*NZ6 + k*NX6 + i + 3*NX6*NZ6 + 3*NX6 + 3;

            u[index] = u2[idx];
            v[index] = v2[idx];
            w[index] = w2[idx];

        }}}

        printf("22222222222\n");

        for( k = 0; k < NZ6;  k++ ){
        for( i = 0; i < NX6;  i++ ){
            int index,j;
            for( j = 0; j < 3; j++ ){
                index = j*NX6*NZ6 + k*NX6 + i;
                u[index] = u[index + (NYD6-7)*NX6*NZ6];
                v[index] = v[index + (NYD6-7)*NX6*NZ6];
                w[index] = w[index + (NYD6-7)*NX6*NZ6];
            }

            for( j = 0; j < 3; j++ ){
                index = (NYD6-1-j)*NX6*NZ6 + k*NX6 + i;
                u[index] = u[(6-j)*NX6*NZ6 + k*NX6 + i];
                v[index] = v[(6-j)*NX6*NZ6 + k*NX6 + i];
                w[index] = w[(6-j)*NX6*NZ6 + k*NX6 + i];
            }
        }}

        printf("22222222222\n");

        for( k = 0; k < NZ6;  k++ ){
        for( j = 0; j < NYD6; j++ ){
            int index, i;
            for( i = 0; i < 3; i++ ){
                index = j*NX6*NZ6 + k*NX6 + i;
                u[index] = u[index + NX6-7];
                v[index] = v[index + NX6-7];
                w[index] = w[index + NX6-7];
            }

            for( i = 0; i < 3; i++ ){
                index = j*NX6*NZ6 + k*NX6 + NX6-1-i;
                u[index] = u[index - NX6+7];
                v[index] = v[index - NX6+7];
                w[index] = w[index - NX6+7];
            }

        }}

        printf("22222222222\n");

        double *f[19];
        for( i = 0; i < 19; i++ ){
            f[i] = (double*)malloc(NX6*NYD6*NZ6*sizeof(double));
            memset(f[i], 0.0, NX6*NYD6*NZ6*sizeof(double));
        }

        double e[19][3]={{0.0,0.0,0.0},{1.0,0.0,0.0},{-1.0,0.0,0.0},{0.0,1.0,0.0},{0.0,-1.0,0.0},{0.0,0.0,1.0},{0.0,0.0,-1.0},
					{1.0,1.0,0.0},{-1.0,1.0,0.0},{1.0,-1.0,0.0},{-1.0,-1.0,0.0},{1.0,0.0,1.0},{-1.0,0.0,1.0},{1.0,0.0,-1.0},
					{-1.0,0.0,-1.0},{0.0,1.0,1.0},{0.0,-1.0,1.0},{0.0,1.0,-1.0},{0.0,-1.0,-1.0}};
        double W[19]={(1.0/3.0),(1.0/18.0),(1.0/18.0),(1.0/18.0),(1.0/18.0),(1.0/18.0),(1.0/18.0),(1.0/36.0),(1.0/36.0)
				  ,(1.0/36.0),(1.0/36.0),(1.0/36.0),(1.0/36.0),(1.0/36.0),(1.0/36.0),(1.0/36.0),(1.0/36.0),(1.0/36.0)
				  ,(1.0/36.0)};
        double udot;

        for( k = 0; k < NZ6;  k++ ) {
        for( j = 0; j < NYD6; j++ ) {
        for( i = 0; i < NX6;  i++ ) {
            const int index = j*NX6*NZ6 + k*NX6 + i;
            double rho = 1.0;
            udot = u[index]*u[index] + v[index]*v[index] + w[index]*w[index];

            f[0][index] = W[0]*rho*(1.0-1.5*udot);
            for( int dir = 1; dir <= 18; dir++ ) {
                f[dir][index] = W[dir] * rho *( 1.0 + 3.0 *( e[dir][0] * u[index] + e[dir][1] * v[index] + e[dir][2] * w[index])
                                    + 4.5 *( e[dir][0] * u[index] + e[dir][1] * v[index] + e[dir][2] * w[index] )
                                    *( e[dir][0] * u[index] + e[dir][1] * v[index] + e[dir][2] * w[index] )- 1.5*udot );
            }
        }}}

        for( i = 0; i < 19; i++ ){
            char path[10];
            sprintf( path, "f%d", i );
            writeData( f[i], path, myid, NX6*NYD6*NZ6 );
        }

        writeData( u, "u", myid, NX6*NYD6*NZ6 );
        writeData( v, "v", myid, NX6*NYD6*NZ6 );
        writeData( w, "w", myid, NX6*NYD6*NZ6 );

        double *rho;
        rho = (double*)malloc(NX6*NYD6*NZ6*sizeof(double));
        for( i = 0; i < NX6*NYD6*NZ6; i++) rho[i] = 1.0;

        writeData( rho, "rho", myid, NX6*NYD6*NZ6 );

        free(rho);

        for( i = 0; i < 19; i++ ){
            free(f[i]);
        }
    }



    free(u);
    free(v);
    free(w);
}
