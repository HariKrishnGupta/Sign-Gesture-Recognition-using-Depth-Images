#include<iostream>
#include<vector>
#include<cstdio>
#include<algorithm>
#include<sstream>
#include<fstream>
#include<cstdlib>
#include<cmath>
#include<time.h>
using namespace std;

#define K 195
#define DEBUG
#define EACHITER
#define inf HUGE_VAL
#define tau 1e-12
#define eps 1e-3
#define IUP(t) (y[t] == 1 && alpha[t]<C) || (y[t] ==-1 && alpha[t]>0)
#define ILOW(t) (y[t] == 1 && alpha[t]>0) || (y[t] ==-1 && alpha[t]<C)
#define IUPEQ(t) (y[t] == 1 && alpha[t]==C) || (y[t] ==-1 && alpha[t]==0)
#define ILOWEQ(t) (y[t] == 1 && alpha[t]==0) || (y[t] ==-1 && alpha[t]==C)
#define PI 3.14159
#define lambda 1.0
double C = 1.0;
double r;
int D;
int SEGMENTS=3,SD=1;
int nSV=0;
ofstream fout1("consensus.txt");
vector<int> knn(vector< vector<double> >&, vector<int>&,vector<int>&,vector<int>&,vector<int>&);
vector<int> knn(vector< vector<double> >&, vector< vector<double> >&,vector<int>&,vector<int>&,vector<int>&);
vector< vector<double> > kmeans(vector< vector<double> >&, vector< vector<double> >&,vector<int>&);
vector<int> selectB(vector<double> &,vector<double> &,vector<int> &,vector< vector<double> > &);
vector<double> SMO(vector< vector<double> > ,vector<int> &);
vector< vector<double> > calcW(int m,int n,double gamma);
vector< vector<double> > randu(int m,int n);
vector< vector<double> > repmat(vector< vector<double> >,int m,int n);
template <class T> void print2D(string var,vector< vector<T> > &mat);
template <class T> void print1D(string var,vector<T> &mat);
template <class T> double mean(vector<T> &x){
	double sum=0.0;
	for(int i=0;i<x.size();i++){
		sum+=x[i];
	}
	double m = sum/x.size();
	return m;
}
template <class T> double stddev(vector<T> &x){
	double sqsum = 0.0;
	for(int i=0;i<x.size();i++){
		sqsum += x[i]*x[i];
	}
	double fterm = sqsum/x.size();
	double m = mean(x);
	double sterm = m*m;
	double sd = sqrt(fterm-sterm);
	return sd;
}
template <class T> double mean(T* x,int len){
	double sum=0.0;
	for(int i=0;i<len;i++){
		sum+=x[i];
	}
	double mean = sum/(double)len;
	return mean;
}
template <class T> double stddev(T *x,int len){
	double sqsum = 0.0;
	for(int i=0;i<len;i++){
		sqsum += x[i]*x[i];
	}
	double fterm = sqsum/len;
	double m = mean(x,len);
	double sterm = m*m;
	double sd = sqrt(fterm-sterm);
	return sd;
}

bool myfunction (double i, double j){ return i>=j;}
void getData(string,vector< vector<double> > &,vector<int>&);
void getDataY(string,vector<int> &);
double getK(vector<double> &r1, vector<double> &r2){
	double sum=0;
	for(int i=0;i<r1.size();i+=1)
		sum+=(r1[i]*r2[i]);
	//sum =sum/D;
	return sum;                                                     
}
template <class T> void print1D(string var,vector<T> &mat){
	cout<<var<<endl;
	for(int i=0;i<mat.size();i++){
		cout<<mat[i]<<" ";
	}
	cout<<endl;
}
template <class T> void print1D(string var,T *mat,int len){
	cout<<var<<endl;
	for(int i=0;i<len;i++){
		cout<<mat[i]<<" ";
	}
	cout<<endl;
}
template <class T> void print2D(string var,vector< vector<T> > &mat){
	cout<<var<<endl;
	for(int i=0;i<mat.size();i++){
		for(int j=0;j<mat[i].size();j++){
			cout<<mat[i][j]<<" ";
		}
		cout<<endl;
	}
}
double findObj(vector<double> &alpha,vector< vector<double> > &Q){
	double obj=0.0;
	for(int i=0;i<alpha.size();i++){
		for(int j=0;j<alpha.size();j++){
			obj+=alpha[i]*alpha[j]*Q[i][j];
		}
		obj-=alpha[i];
	}
	return obj;
}
int main(){
	srand(time(0));
	vector< vector<double> > X;
	vector<int> y;
        getData(string("test1.txt"),X,y);
	long int len = X.size();
	long int featureCount = X[0].size();
	D = featureCount;
	//int j,g;
	vector<int>SVM(len,-1);
	int i,j;int m;
	// initial centroid for kmeans
	if(len>=2000){
		int start,end;
		int t=30;
		for(int count=0;count<1;count++){
			//t+=10;
			vector< vector<double> >initialCentroid(t, vector<double>(featureCount,0.0));
			i=0;
			vector<int>clusterIndex(t,-1);
			int l=0;
			while(t--){
				m = rand()%(len-1);
				clusterIndex[l++] = m;
				for(int h=0;h<featureCount;h++){
					initialCentroid[i][h] = X[m][h];
				}
				i++;
			}
			initialCentroid = kmeans(X,initialCentroid,y);
			cout<<"k means done "<<endl;
			i=0;
			start = clock();
			int validationPoints = initialCentroid.size();
			int trainingPoints = len - validationPoints;
			//cout<<"training points = "<<trainingPoints<<"\t"<<"validation points = "<<validationPoints<<endl;
			vector<int>trainingData(trainingPoints,0);
			int k=0;int flag=0;
			for(i=0;i<len;i++){
				flag=0;
				for(j=0;j<l;j++){
					if(clusterIndex[j]==i){
						flag=1;
					}
				}
				if(flag==0){
					trainingData[k++] = i;
				}
			}
			cout<<"knn start"<<endl;			
			SVM = knn(X,initialCentroid,trainingData,y,SVM);
		}
		end=clock();
		cout<<"taken for knn = "<<(((double)end-start)/CLOCKS_PER_SEC)<<endl;
	}
	else{
		i=0;
		m=1000;
		int start,end;
		start = clock();
		for(int count=0;count<5;count++){
			//m = rand()%(len/3)+1;
			m+=100;
			int validationPoints = 2*m;
			int trainingPoints = len - validationPoints;
			//cout<<"training points = "<<trainingPoints<<"\t"<<"validation points = "<<validationPoints<<endl;
			vector<int>trainingData(trainingPoints,0);
			vector<int>validationData(validationPoints,0);
			int k=0,l=0;
			for(i=0;i<len;i++){
				if(i<m || i>=(len-m)){
					validationData[k]=i;
					k++;
				}
				else{
					trainingData[l]=i;
					l++;
				}
			
			}
			SVM = knn(X,validationData,trainingData,y,SVM);
		}
		end=clock();
		cout<<"taken for knn = "<<(((double)end-start)/CLOCKS_PER_SEC)<<endl;
	}
	fout1.close();
	/*** Consensus KNN SVM ***/
	int start,end;
	vector< vector<double> > cX;
	vector<int> cy;	
	getData(string("consensus.txt"),cX,cy);
	len = cX.size();
	featureCount = cX[0].size();
	vector<double> alphaDATA(cX.size(),-1);
	start = clock();
	alphaDATA=SMO(cX,cy);
	end = clock();
	cout<<"time taken for consensus = "<<(((double)end-start)/CLOCKS_PER_SEC)<<endl;
	int positiveSV=0;
	int negativeSV=0;
	for(int t = 0; t<len;t++){
		if(alphaDATA[t] > 0){
			if(alphaDATA[t]*cy[t]>0){
				positiveSV++;
			}
			else if(alphaDATA[t]*cy[t]<0)
				negativeSV++;
		}
	}
	cout<<"number of support vector = "<<(positiveSV+negativeSV)<<endl;
	ofstream fout;
	string filename = "consensus.model";
	fout.open(filename.c_str(),ios::out);	
	fout<<"svm_type c_svc"<<endl;
	fout<<"kernel_type linear"<<endl;
	//fout<<"gamma 0.5"<<endl;
	fout<<"nr_class 2"<<endl;
	fout<<"total_sv "<<(positiveSV+negativeSV)<<endl;
	fout<<"rho "<<r<<endl;
	if(y[0]==1)
		fout<<"label 1 -1"<<endl;
	else
		fout<<"label -1 1"<<endl;
	fout<<"nr_sv "<<positiveSV<<" "<<negativeSV<<endl;
	fout<<"SV"<<endl;
	for(int t=0;t<cX.size();t++){
		if(alphaDATA[t]>0){
			fout<<alphaDATA[t] * cy[t];
			for(int u=0;u<cX[0].size();u++){
				fout<<" "<<(u+1)<<":"<<cX[t][u];
			}
			fout<<endl;
		}		
	}
	fout.close();
	return 0;
}
vector<double> SMO(vector< vector<double> > X,vector<int> &y){
	int len = y.size(); //number of examples
	//int m = featureWeight.size(); //number of features
	vector< vector<double> > Q(len,vector<double>(len,0));
#ifdef DEBUG
	//print2D("newX in SMO = ",X);
#endif
	for(int i=0;i<len;i++){
		for(int j=0;j<len;j++){
			Q[i][j] = y[i]*y[j]*getK(X[i],X[j]);
		}
	}
#ifdef DEBUG
	//print2D("Q in SMO = ",Q);
#endif
	vector<double> alpha(len,0.0);
	vector<double> gradient(len,-1.0);
	int count=0;
	while(true){
		vector<int> working_set = selectB(alpha,gradient,y,Q);
		int i=working_set[0];
		int j=working_set[1];
		if(j==-1) break;
		//cout<<"Working set in iter "<<count+1<<" = ("<<i<<","<<j<<")"<<endl;

		//working set is i=working_set[0] and j=working_set[1]
		double quad_coef = Q[i][i] + Q[j][j] -2*y[i]*y[j]*Q[i][j];
		if(quad_coef<=0)	quad_coef = tau;
		double b = -y[i]*gradient[i] + y[j]*gradient[j];

		double oldAi = alpha[i];
		double oldAj = alpha[j];
		alpha[i] += y[i]*b/quad_coef;
		alpha[j] -= y[j]*b/quad_coef;

		//project alpha back
		double sum = y[i]*oldAi + y[j]*oldAj;
		if(alpha[i]>C) alpha[i]=C;
		if(alpha[i]<0) alpha[i]=0;
		alpha[j] = y[j]*(sum-y[i]*alpha[i]);
		if(alpha[j]>C) alpha[j]=C;
		if(alpha[j]<0) alpha[j]=0;
		alpha[i] = y[i]*(sum-y[j]*alpha[j]);

		//update gradient
		double deltaAi = alpha[i] - oldAi;
		double deltaAj = alpha[j] - oldAj;
		for(int t=0;t<len;t++)
			gradient[t] +=Q[t][i]*deltaAi+Q[t][j]*deltaAj;
		count++;
	}
	
		/* calculating rho*/ 
		
		int inRangeAlphaCount = 0;
		double ub = inf, lb = -inf, sumInRange = 0;
		for(int t=0;t<len;t++){
			double yG = y[t]*gradient[t];
			if(IUPEQ(t))
			{
				if(lb>yG)
					lb = lb;
				else
					lb = yG;
			}
			else if(ILOWEQ(t))
			{
				if(ub<yG){
					ub = ub;
				}
				else
					ub = yG;
			}
			else
			{
				++inRangeAlphaCount;
				sumInRange += yG;
			}
		}
		if(inRangeAlphaCount>0)
			r = sumInRange/inRangeAlphaCount;
		else
			r = (ub+lb)/2;
		
	return alpha;
}
vector<int> knn(vector< vector<double> >&X,vector<int>&validationData,vector<int>&trainingData,vector<int>&y,vector<int>&SVM){
	int i,j,k;int l,m;
	int len = X.size();
	int featureCount = X[0].size();
	double dist=0.0;
	double disttemp,temp;
	//calculate K nearest neighbours
	int targetNeighbours = K;
	double kNeighbours[targetNeighbours][featureCount];
	int nValidation = validationData.size();
	int nTraining = trainingData.size();
	//cout<<"inside knn - \n"<<"len = "<<len<<"featureCount = "<<featureCount<<endl;
	vector< vector<double> >neighbourMatrix(nValidation, vector<double>(targetNeighbours,0.0));
	int var=0;int g;
	for(i=0;i<nValidation;i++){
		for(g=0;g<targetNeighbours;g++){
			kNeighbours[g][0]=0;
			kNeighbours[g][1]=100000.00;
		}
		for(j=0;j<nTraining;j++){
			disttemp=0.0;
			for(k=0;k<featureCount;k+=1){
				temp = (X[validationData[i]][k]-X[trainingData[j]][k]);
				//temp = (X[validationData[i]][k+1]-X[trainingData[j]][k+1]);
				disttemp = disttemp + (temp*temp); 
			}
			dist = sqrt(disttemp);
			for(l=0;l<targetNeighbours;l++){
				if(dist <= kNeighbours[l][1]){
				 	for(m=targetNeighbours-1;m>l;m--){
				 		kNeighbours[m][0]=kNeighbours[m-1][0];
				 		kNeighbours[m][1]=kNeighbours[m-1][1];
				 	}
				 	kNeighbours[l][0]=trainingData[j];
				 	kNeighbours[l][1] = dist;
				 	break;
				 }		
			}			
		}
		for(l=0;l<targetNeighbours;l++){
			neighbourMatrix[var][l] = kNeighbours[l][0];
		}
		var++;
	}
	for(k=0;k<nValidation;k++){	
		vector< vector<double> >tempDataMatrix(targetNeighbours, vector<double>(featureCount,0.0));
		vector<int> tempY(targetNeighbours);
		for(i=0;i<targetNeighbours;i++){
			for(j=0;j<featureCount;j++){
				tempDataMatrix[i][j] = X[neighbourMatrix[k][i]][j];
			}
			tempY[i] = y[neighbourMatrix[k][i]];
		}
		vector<double> alphaDATA(targetNeighbours,-1);
		alphaDATA = SMO(tempDataMatrix,tempY);
		#ifdef EACHITER
			//print1D("alphaDATA = ",alphaDATA);
		#endif
		int sv=0;
		int positiveSV=0;
		int negativeSV=0;
		int d=0;
		for(int t = 0; t<targetNeighbours;t++){
			if(alphaDATA[t] > 0){
				if(alphaDATA[t]*tempY[t]>0){
					positiveSV++;
				}
				else if(alphaDATA[t]*tempY[t]<0)
					negativeSV++;
			}
		}
		/*
		string str = "knn";
		stringstream oss;
		oss<<str<<k<<".model"<<"\0";
		string s = oss.str();
		ofstream fout;
		fout.open(s.c_str(),ios::out);
		fout<<"svm_type c_svc"<<endl;
		fout<<"kernel_type linear"<<endl;
		fout<<"nr_class 2"<<endl;
		fout<<"total_sv "<<(positiveSV+negativeSV)<<endl;
		fout<<"rho "<<r<<endl;
		fout<<"label -1 1"<<endl;
		fout<<"nr_sv "<<positiveSV<<" "<<negativeSV<<endl;
		fout<<"SV"<<endl;
		for(int t=0;t<targetNeighbours;t++){
			if(alphaDATA[t]>0){
				fout<<alphaDATA[t] * tempY[t];
				for(int u=0;u<featureCount;u++){
					fout<<" "<<(u+1)<<":"<<tempDataMatrix[t][u];
				}
				fout<<endl;
			}
			
		}	
		fout.close();
		*/
		for(i=0;i<targetNeighbours;i++){
			if(alphaDATA[i]>0){
				if(SVM[neighbourMatrix[k][i]]==-1){
					nSV+=1;
					SVM[neighbourMatrix[k][i]]=1;
					fout1<<y[neighbourMatrix[k][i]];
					for(int t=0;t<featureCount;t++){
						fout1<<" "<<X[neighbourMatrix[k][i]][t];
					}
					fout1<<endl;
				}
			}
		}
	}
	
	return SVM;
}

vector<int> knn(vector< vector<double> >&X,vector< vector<double> >&initialCentroid,vector<int>&trainingData,vector<int>&y,vector<int>&SVM){
	int i,j,k;int l,m;
	int len = X.size();
	int featureCount = X[0].size();
	double dist=0.0;
	double disttemp,temp,temp1;
	//calculate K nearest neighbours
	int targetNeighbours = K;
	double kNeighbours[targetNeighbours][2];
	int nValidation = initialCentroid.size();
	int nTraining = trainingData.size();
	vector< vector<double> >neighbourMatrix(nValidation, vector<double>(targetNeighbours,0.0));
	int var=0;int g;
	//cout<<"inside knn - \n"<<"len = "<<len<<"featureCount = "<<featureCount<<endl;
	for(i=0;i<nValidation;i++){
		for(g=0;g<targetNeighbours;g++){
			kNeighbours[g][0]=0;
			kNeighbours[g][1]=100000.00;
		}
		for(j=0;j<nTraining;j++){
			disttemp=0.0;
			for(k=0;k<featureCount;k+=1){
				temp = (initialCentroid[i][k]-X[trainingData[j]][k]);
				//temp = (initialCentroid[i][k+1]-X[trainingData[j]][k+1]);
				disttemp = disttemp + (temp*temp); 
			}
			dist = sqrt(disttemp);
			for(l=0;l<targetNeighbours;l++){
				if(dist <= kNeighbours[l][1]){
				 	for(m=targetNeighbours-1;m>l;m--){
				 		kNeighbours[m][0]=kNeighbours[m-1][0];
				 		kNeighbours[m][1]=kNeighbours[m-1][1];
				 	}
				 	kNeighbours[l][0]=trainingData[j];
				 	kNeighbours[l][1] = dist;
				 	break;
				 }		
			}			
		}
		for(l=0;l<targetNeighbours;l++){
			neighbourMatrix[var][l] = kNeighbours[l][0];
		}
		var++;
	}
	int neigh;
	for(k=0;k<nValidation;k++){	
		vector< vector<double> >tempDataMatrix(targetNeighbours, vector<double>(featureCount,0.0));
		vector<int> tempY(targetNeighbours);
		for(i=0;i<targetNeighbours;i++){
			for(j=0;j<featureCount;j++){
				tempDataMatrix[i][j] = X[neighbourMatrix[k][i]][j];
			}
			tempY[i] = y[neighbourMatrix[k][i]];
		}
		//cout<<"alpha \n";
		vector<double> alphaDATA(targetNeighbours,-1);
		alphaDATA = SMO(tempDataMatrix,tempY);
		#ifdef EACHITER
			//print1D("alphaDATA = ",alphaDATA);
		#endif
		/*
		string str = "knn";
		stringstream oss;
		oss<<str<<k<<".model"<<"\0";
		string s = oss.str();
		ofstream fout;
		fout.open(s.c_str(),ios::out);
		fout<<"svm_type c_svc"<<endl;
		fout<<"kernel_type linear"<<endl;
		fout<<"nr_class 2"<<endl;
		fout<<"total_sv "<<(positiveSV+negativeSV)<<endl;
		fout<<"rho "<<r<<endl;
		fout<<"label -1 1"<<endl;
		fout<<"nr_sv "<<positiveSV<<" "<<negativeSV<<endl;
		fout<<"SV"<<endl;
		for(int t=0;t<targetNeighbours;t++){
			if(alphaDATA[t]>0){
				fout<<alphaDATA[t] * tempY[t];
				for(int u=0;u<featureCount;u++){
					fout<<" "<<(u+1)<<":"<<tempDataMatrix[t][u];
				}
				fout<<endl;
			}
			
		}	
		fout.close();
		*/
		for(i=0;i<targetNeighbours;i++){
			if(alphaDATA[i]>0){
				if(SVM[neighbourMatrix[k][i]]==-1){
					nSV+=1;
					SVM[neighbourMatrix[k][i]]=1;
					fout1<<y[neighbourMatrix[k][i]];
					for(int t=0;t<featureCount;t++){
						fout1<<" "<<X[neighbourMatrix[k][i]][t];
					}
					fout1<<endl;
				}
			}
		}
	}
	
	return SVM;
}
vector< vector<double> > kmeans(vector< vector<double> >&X,vector< vector<double> >&initialCentroid,vector<int>&y){
	
	int i,j,k;
	int len = X.size();
	int featureCount = X[0].size();
	double dist=0.0;
	int nAnchorPoints = initialCentroid.size();
	int iteration = 5;
	double min = 10000000.00;
	double temp;int index;double temp1;
	//cout<<"inside kmeans - \n"<<"len = "<<len<<"\t"<<"feature count = "<<featureCount<<"\tanchor points = "<<nAnchorPoints<<endl;
	vector<int>centroidIndex(len,0);
	vector<int>countPoints(nAnchorPoints,0);
	for(i=0;i<iteration;i++){
		for(j=0;j<len;j++){
			min=100000.00;
			for(k=0;k<nAnchorPoints;k++){
				dist = 0.0;
				for(int h=0;h<featureCount;h+=1){
					temp = (X[j][h] - initialCentroid[k][h]);
					//temp1 = (X[j][h+1] - initialCentroid[k][h+1]);
					temp = temp * temp;
					//temp1 = temp1*temp1;
					dist = dist + temp;
				}
				dist = sqrt(dist);
				if(dist<min){
					min = dist;
					index = k;
				}
			}
			centroidIndex[j] = index;
		}
		for(k=0;k<nAnchorPoints;k++){
			vector<double>sum(featureCount,0.0);
			int no=0;
			for(j=0;j<len;j++){
				if(centroidIndex[j]==k){
					for(int h=0;h<featureCount;h+=2){
						sum[h] = sum[h] + X[j][h]; 
					}
					no++;
				}
			}
			for(int h=0;h<featureCount;h++){
				sum[h] = sum[h]/no;
				initialCentroid[k][h] = sum[h];
			}
			countPoints[k] = no;	
		}
	}
	/*
	cout<< " final centroid matrix -- "<<endl;
	for(k=0;k<nAnchorPoints;k++){
		for(int h=0;h<featureCount;h++){
			cout<<initialCentroid[k][h]<<"\t";
		}
		cout<<"\n";
	}
	cout<<"final index vector -- "<<endl;
	for(i=0;i<len;i++){
		cout<<centroidIndex[i]<<endl;
	}
	* */	
	return initialCentroid;
}


vector<int> selectB(vector<double> &alpha,vector<double> &gradient,vector<int> &y,vector< vector<double> > &Q){
	int i=-1;
	int len = y.size();
	double G_max = -inf;
	double G_min = inf;
	for(int t=0;t<len;t++){
		if( IUP(t)){ //Iup
			if(-y[t]*gradient[t] >= G_max){
				i=t;
				G_max = -y[t]*gradient[t];
			}
		}
	}
	//i j from
	int j=-1;
	double obj_min = inf;
	for(int t=0;t<len;t++){
		if(ILOW(t)){//Ilow
			double grad_diff = G_max+y[t]*gradient[t];
			if(-y[t]*gradient[t]<=G_min) G_min = -y[t]*gradient[t];
			if(grad_diff>0){
				double quad_coef = Q[i][i]+Q[t][t]-2.0*y[i]*y[t]*Q[i][t];
				if(quad_coef<=0) quad_coef = tau;
				if(-(grad_diff*grad_diff)/quad_coef <=obj_min){
					j=t;
					obj_min = -(grad_diff*grad_diff)/quad_coef;
				}
			}
		}
	}
	if(G_max - G_min < eps) return vector<int>(2,-1);
	vector<int> out(2);
	out[0]=i;
	out[1]=j;
	return out;
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
                        if(yl==0)
							yl=-1;
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
