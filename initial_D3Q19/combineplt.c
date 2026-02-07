#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include "variables.h"

double *x , *y , *z ;
double *u_global, *v_global, *w_global, *rho_global;

void ReadData(
    double *arr_h,
    const char *folder,     const char *fname,      const int myid  )
{
    char path[100];
    sprintf( path, "./%s/%s_%d.bin", folder, fname, myid );

    FILE *data;
    if((data = fopen(path, "rb")) == NULL) {
        printf("Read data error, exit...\n");
        //CHECK_MPI( MPI_Abort(MPI_COMM_WORLD, 1) );
    }

    fread( arr_h, sizeof(double), NX6*NZ6*NYD6, data );
    fclose( data );
}

void Mesh_scan() {
    int i,j,k;
    FILE *meshZ, *meshX, *meshY;

    meshZ = fopen("meshZ.DAT","r");
    for( k = 0; k < NZ6; k++ ){
        fscanf( meshZ, "%lf", &z[k] );
    }
    fclose(meshZ);

    meshX = fopen("meshX.DAT","r");
    for( i = 0; i < NX6; i++ ){
        fscanf( meshX, "%lf", &x[i] );
    }
    fclose(meshX);

    meshY = fopen("meshY.DAT","r");
    for( j = 0; j < NY6; j++ ){
        fscanf( meshY, "%lf", &y[j] );
    }
    fclose(meshY);
}


void wirte_ASCII_of_str(char * str, FILE *file);

void Output3Dvelocity(){

    char filename_E2[300];

    sprintf(filename_E2, "%dx%dx%d.plt", (NX6-6), (NY6-6), (NZ6-6));
    printf("%s\n", filename_E2);

    FILE *fpE3;

    fpE3 = fopen(filename_E2, "wb");

    int IMax = NX6-6;

    int JMax = NY6-6;

    int KMax = NZ6-6;

    char Title[] = "Particle intensity";

    char Varname1[] = "X";

    char Varname2[] = "Y";

    char Varname3[] = "Z";

    char Varname4[] = "Vel";

    char Varname5[] = "U";

	char Varname6[] = "V";

	char Varname7[] = "W";

    char Zonename1[] = "Zone 001";

    float ZONEMARKER = 299.0;

    float EOHMARKER = 357.0;

    //==============Header Secontion =================//
    //------1.1 Magic number, Version number
    char MagicNumber[] = "#!TDV101";
    //cout << "watchout" << sizeof(MagicNumber) << endl;
    fwrite(MagicNumber, 8, 1, fpE3);

    //---- - 1.2.Integer value of 1.----------------------------------------------------------
    int IntegerValue = 1;
    fwrite(&IntegerValue, sizeof(IntegerValue), 1, fpE3);

    //---- - 1.3.Title and variable names.------------------------------------------------ -
    //---- - 1.3.1.The TITLE.
    wirte_ASCII_of_str(Title, fpE3);

    //---- - 1.3.2 Number of variables(NumVar) in the c_strfile.
    int NumVar = 7;
    fwrite(&NumVar, sizeof(NumVar), 1, fpE3);

    //------1.3.3 Variable names.N = L[1] + L[2] + ....L[NumVar]
    wirte_ASCII_of_str(Varname1, fpE3);
    wirte_ASCII_of_str(Varname2, fpE3);
    wirte_ASCII_of_str(Varname3, fpE3);
    wirte_ASCII_of_str(Varname4, fpE3);
    wirte_ASCII_of_str(Varname5, fpE3);
    wirte_ASCII_of_str(Varname6, fpE3);
	wirte_ASCII_of_str(Varname7, fpE3);
    //---- - 1.4.Zones------------------------------------------------------------------ -
    //--------Zone marker.Value = 299.0
    fwrite(&ZONEMARKER, 1, sizeof(ZONEMARKER), fpE3);

    //--------Zone name.
    wirte_ASCII_of_str(Zonename1, fpE3);

    //--------Zone color
    int ZoneColor = -1;
    fwrite(&ZoneColor, sizeof(ZoneColor), 1, fpE3);

    //--------ZoneType
    int ZoneType = 0;
    fwrite(&ZoneType, sizeof(ZoneType), 1, fpE3);

    //--------DaraPacking 0=Block, 1=Point
    int DaraPacking = 1;
    fwrite(&DaraPacking, sizeof(DaraPacking), 1, fpE3);

    //--------Specify Var Location. 0 = Don't specify, all c_str is located at the nodes. 1 = Specify
    int SpecifyVarLocation = 0;
    fwrite(&SpecifyVarLocation, sizeof(SpecifyVarLocation), 1, fpE3);

    //--------Number of user defined face neighbor connections(value >= 0)
    int NumOfNeighbor = 0;
    fwrite(&NumOfNeighbor, sizeof(NumOfNeighbor), 1, fpE3);

    //-------- - IMax, JMax, KMax
    fwrite(&IMax, sizeof(IMax), 1, fpE3);
    fwrite(&JMax, sizeof(JMax), 1, fpE3);
    fwrite(&KMax, sizeof(KMax), 1, fpE3);

    //----------// -1 = Auxiliary name / value pair to follow 0 = No more Auxiliar name / value pairs.
    int AuxiliaryName = 0;
    fwrite(&AuxiliaryName, sizeof(AuxiliaryName), 1, fpE3);

    //----I HEADER OVER--------------------------------------------------------------------------------------------

    //=============================Geometries section=======================
    //=============================Text section======================
    // EOHMARKER, value = 357.0
    fwrite(&EOHMARKER, sizeof(EOHMARKER), 1, fpE3);

    //================II.Data section===============//
    //------ 2.1 zone---------------------------------------------------------------------- -
    fwrite(&ZONEMARKER, sizeof(ZONEMARKER), 1, fpE3);

    //--------variable c_str format, 1 = Float, 2 = Double, 3 = LongInt, 4 = ShortInt, 5 = Byte, 6 = Bit
    int fomat1 = 2;
    int fomat2 = 2;
    int fomat3 = 2;
    int fomat4 = 2;
    int fomat5 = 2;
    int fomat6 = 2;
	int fomat7 = 2;
    fwrite(&fomat1, sizeof(fomat1), 1, fpE3);
    fwrite(&fomat2, sizeof(fomat2), 1, fpE3);
    fwrite(&fomat3, sizeof(fomat3), 1, fpE3);
    fwrite(&fomat4, sizeof(fomat4), 1, fpE3);
    fwrite(&fomat5, sizeof(fomat5), 1, fpE3);
    fwrite(&fomat6, sizeof(fomat6), 1, fpE3);
	fwrite(&fomat7, sizeof(fomat7), 1, fpE3);
    //--------Has variable sharing 0 = no, 1 = yes.
    int HasVarSharing = 0;
    fwrite(&HasVarSharing, sizeof(HasVarSharing), 1, fpE3);

    //----------Zone number to share connectivity list with(-1 = no sharing).
    int ZoneNumToShareConnectivity = -1;
    fwrite(&ZoneNumToShareConnectivity, sizeof(ZoneNumToShareConnectivity), 1, fpE3);

    //----------Zone c_str.Each variable is in c_str format asspecified above.
    int i , j , k;
    for (k = 3; k < NZ6-3; k++){
    for (j = 3; j < NY6-3; j++){
    for (i = 3; i < NX6-3; i++){

        double VarToWrite1 = x[i];
        double VarToWrite2 = y[j];
        double VarToWrite3 = z[k];

        int idx_global = j*NX6*NZ6 + (k)*NX6 + i;

	    double Uvelocity = u_global[idx_global];
	    double Vvelocity = v_global[idx_global];
	    double Wvelocity = w_global[idx_global];
        //double Rho       = rho_global[idx_global];
        double Velocity = pow( (Uvelocity*Uvelocity + Vvelocity*Vvelocity + Wvelocity*Wvelocity), 0.5);

        fwrite(&VarToWrite1, sizeof(VarToWrite1), 1, fpE3);
        fwrite(&VarToWrite2, sizeof(VarToWrite2), 1, fpE3);
        fwrite(&VarToWrite3, sizeof(VarToWrite3), 1, fpE3);

        //fwrite(&Rho,       sizeof(Rho),       1, fpE3);
        fwrite(&Velocity,  sizeof(Velocity),  1, fpE3);
	    fwrite(&Uvelocity, sizeof(Uvelocity), 1, fpE3);
	    fwrite(&Vvelocity, sizeof(Vvelocity), 1, fpE3);
	    fwrite(&Wvelocity, sizeof(Wvelocity), 1, fpE3);

    }}}

    fclose(fpE3);

}


int main (int argc, char *argv[])
{
    int i , j , k , n, myid;
    double *Phys[4];
    /*allocate memory*/
    u_global   = (double*)malloc(NX6*NY6*NZ6*sizeof(double));
    v_global   = (double*)malloc(NX6*NY6*NZ6*sizeof(double));
    w_global   = (double*)malloc(NX6*NY6*NZ6*sizeof(double));
    rho_global = (double*)malloc(NX6*NY6*NZ6*sizeof(double));
    if( u_global == NULL ){
        printf("Memory error, exit...\n");
        return 0;
    }

    x = (double*)malloc( NX6*sizeof(double) );
    y = (double*)malloc( NY6*sizeof(double) );
    z = (double*)malloc( NZ6*sizeof(double)  );

    for( n = 0; n < 4; n++ ){
        Phys[n] = (double*)malloc(NX6*NYD6*NZ6*sizeof(double));
    }

    Mesh_scan();
    
    double Utau = Retau*niu/LZ*2.0;

    //Scan data//
    for( myid = 0 ; myid < jp ; myid++ ){
        printf("Reading data... myid = %d\n", myid);

        ReadData( Phys[0], "result", "u", myid );
        ReadData( Phys[1], "result", "v", myid );
        ReadData( Phys[2], "result", "w", myid );
        ReadData( Phys[3], "result", "rho", myid );
        for( k = 3; k < NZ6-3 ;  k++){
        for( j = 3; j < NYD6-3; j++){
        for( i = 3; i < NX6-3 ; i++){
            int j_global = myid*(NYD6-7) + j ;
            int idx_local = j*NX6*NZ6 + k*NX6 + i;
            int idx_global = j_global*NX6*NZ6 + k*NX6 + i;

            u_global[idx_global] = Phys[0][idx_local] / Utau;
            v_global[idx_global] = Phys[1][idx_local] / Utau;
            w_global[idx_global] = Phys[2][idx_local] / Utau;
            rho_global[idx_global] = Phys[3][idx_local];
        }}}
    }

    free ( Phys[0] );
    free ( Phys[1] );
    free ( Phys[2] );
    free ( Phys[3] );
    //Output data
	Output3Dvelocity();

    free (u_global);
    free (v_global);
    free (w_global);
    free (rho_global);
    return 0;
}

void wirte_ASCII_of_str(char * str, FILE * file)
{
    int value = 0;

    while ((*str) != '\0'){
        value = (int)*str;
        fwrite(&value, sizeof(int), 1, file);
        str++;
    }

    char null_char[] = "";

    value = (int)*null_char;

    fwrite(&value, sizeof(int), 1, file);
}

