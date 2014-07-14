#include<iostream>
#include<vector>
#include<cstdio>
#include<algorithm>
#include<sstream>
#include<fstream>
#include<cstdlib>
#include<cmath>
#include <time.h>
using namespace std;

#define eps 1e-3
#define PI 3.14159
#define lambda 1.0

void getData(string,vector< vector<double> > &,vector<int>&);
vector< vector<double> > calcW(int m,int n,double gamma);
vector< vector<double> > randu(int m,int n);
vector< vector<double> > repmat(vector< vector<double> >,int m,int n);

int main(){

	vector< vector<double> > X;
	vector<int> y;
	string s;
	int targetFeatures;
	cout<<"Enter the dataset filename: ";
	cin>>s;
	getData(s,X,y);
	int len = X.size();
	int featureCount = X[0].size();
	int i,j,k;
	double gamma = 0.1;
	
	cout<<"Number of sample = "<<len<<"\t"<<"Number of features = "<<featureCount<<endl;
	/*
	for(i=0;i<len;i++){
		for(j=0;j<featureCount;j++){
			cout<<X[i][j]<<"\t";
		}
		cout<<y[i];
		cout<<endl;
	}
	*/	
	cout<<"Enter the no. of target features: ";	
	cin>>targetFeatures;
	cout<<"Enter the output dataset filename: ";
	cin>>s;
	int start,end;
	start=clock();
	cout<<"Calculating W ....."<<endl;
	vector< vector<double> >W(featureCount, vector<double>(targetFeatures,0.0));
	vector< vector<double> >bias(1,vector<double>(targetFeatures,0.0));
	vector< vector<double> >rep(len,vector<double>(targetFeatures,0.0));
	W = calcW(featureCount, targetFeatures, gamma);
	bias = randu(1,targetFeatures);
	rep = repmat(bias,len,1);
	/*	
	for(i=0;i<featureCount;i++){
		for(j=0;j<targetFeatures;j++){
			W[i][j] = sqrt(2) * U[i][j];
			cout<<W[i][j]<<"\t";
		}
		cout<<endl;
	}
	*/

	cout<<"Computing G ......"<<endl;
	vector< vector<double> >G(len, vector<double>(targetFeatures,0.0));
	for(i=0;i<len;i++){
		for(j=0;j<targetFeatures;j++){
			G[i][j]=0.0;
			for(k=0;k<featureCount;k++){
				G[i][j] = G[i][j] + X[i][k]*W[k][j];
			}
			G[i][j] = cos(G[i][j]+rep[i][j])/(sqrt(targetFeatures));
		}
	}	
	end=clock();
	cout<<"Time taken to generate fourier features: "<<(((double)end-start)/CLOCKS_PER_SEC)<<endl;	
	ofstream fout;
	fout.open(s.c_str(),ios::out);
	for(i=0;i<len;i++){
		fout<<y[i]<<" ";
		for(j=0;j<targetFeatures-1;j++)
			fout<<G[i][j]<<" ";
		fout<<G[i][j];
		fout<<endl;
	}	
	fout.close();
	cout<<"Output dataset G stored in file - "<<s<<"\n";
	return 0;
}

vector< vector<double> > repmat( vector< vector<double> > bias,int m,int n){
	int i,j,k,l;
	int rlen = bias.size(),clen = bias[0].size();
	vector< vector <double> >a((m*rlen),vector< double> (n*clen,0.0));
	for(i=0;i<m;i++){
		for(j=0;j<n;j++){
			for(k=0;k<rlen;k++){
				for(l=0;l<clen;l++)
					a[(i*rlen)+k][(j*clen)+l] = bias[k][l];
			}
		}
	}
	return a;
}

vector< vector<double> > randu(int m,int n){
	int i,j;
	vector< vector<double> > u(m, vector<double>(n,0.0));
	for(i=0;i<m;i++){
		for(j=0;j<n;j++)
			u[i][j] = 2*(PI)*( (double)rand() / ((double)(RAND_MAX)+(double)(1)) );
	}
	return u;
}

vector< vector<double> > calcW(int m, int n,double gamma){
	vector<double>u(m*n+1,0.0);
	int i,j,k;
	double rand_max =  double(RAND_MAX),root2g = sqrt(2*gamma);
	for(i=0;i<(m*n+1);i++){
		u[i] = ( (double)rand() / ((double)(RAND_MAX)+(double)(1)) );	
	}
	vector< vector<double> >U(m, vector<double>(n,0.0));
	double pi = 4.0 * atan(1.0);
	double square,amp,angle;
	k=0;
	for(i=0;i<m;i++){
		for(j=0;j<n;j++){
			if(k%2==0){
				square = -2.0*log(u[k]);
				if(square < 0.0){
					square = 0.0;
				}
				amp = sqrt(square);
				angle = 2.0*PI*u[k+1];
				U[i][j] = amp*sin(angle)*root2g;
			}
			else
				U[i][j] = amp*cos(angle)*root2g;
			k++;
		}
	}
	return U;
}

void getData(string filename,vector< vector<double> > &X,vector< int > &y){
	ifstream file;
        file.open(filename.c_str(),ios::in);
        if(file.is_open()){
                string line;
                vector<double> xi;
                while(!file.eof()){
                        getline(file,line);
                        if(file.eof()) break;
                        istringstream iss(line);
                        int yl;
                        iss>>yl;
						if(yl==2) yl=-1;
                        y.push_back(yl);
                        do{
                                double val;
                                iss>>val;
                                xi.push_back(val);
                        }while(!iss.eof());
                        X.push_back(xi);
                        xi.clear();
                }
                file.close();
        }
        else{
                cout<<"Unable to open file "<<filename<<endl;
                exit(1);
        }

}
