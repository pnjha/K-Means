#ifndef DISTANCE_H
#define DISTANCE_H


using namespace std;


double manhattan_distance(vector<int>image1,vector<int>image2,int threads){

  double dist = 0.0;
  // #pragma omp parallel for schedule(static) num_threads(threads) 
  #pragma omp parallel for reduction(+:dist) num_threads(threads)
  for(int i = 0;i<image1.size();i++){

    // #pragma omp atomic
    dist += abs(image1[i] - image2[i]);

  }

  return dist;
}

double chebyshev_distance(vector<int>image1,vector<int>image2,int threads){

  int size = image1.size();

  double dist[size][1] = {0.0};
  // memset(double,0,sizeof(dist[0]));

  //Following cannot be used for OpenMp v3 or lesser
  //  #pragma omp parallel for reduction(max:dist) num_threads(threads)  
  // omp_set_lock() could also be used
  
  #pragma omp parallel for num_threads(threads)
  for(int i = 0;i<image1.size();i++){

    // if(abs(image1[i] - image2[i])>dist)
    	dist[i][0] = abs(image1[i] - image2[i]);

  }
  double maxdist = 0.0;
  for(int i = 0;i<image1.size();i++){
    if(maxdist<dist[i][0])
      maxdist = dist[i][0];
  }

  return maxdist;
}

double hellinger_distance(vector<int>image1,vector<int>image2,int threads){

  double dist = 0.0;
  // #pragma omp parallel for schedule(static) num_threads(threads) 
  #pragma omp parallel for reduction(+:dist) num_threads(threads)
  for(int i = 0;i<image1.size();i++){

    // #pragma omp atomic
    dist += pow((sqrt(image1[i]) - sqrt(image2[i])),2);

  }

  dist = sqrt(dist)/(sqrt(2));

  return dist;
}

double euclidean_distance(vector<int>image1,vector<int>image2,int threads){

  double dist = 0.0;
  // #pragma omp parallel for schedule(static) num_threads(threads) 
  #pragma omp parallel for reduction(+:dist) num_threads(threads)
  for(int i = 0;i<image1.size();i++){

    // #pragma omp atomic
    dist += pow((image1[i] - image2[i]),2);

  }

  dist = sqrt(dist);

  return dist;
}



#endif
