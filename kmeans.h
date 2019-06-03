#ifndef KMEANS_H
#define KMEANS_H

vector<vector<int>> parallel_average_face(vector<vector<int>>& image_pixel,vector<string>& img_info,
                                          vector<string>& avg_face_info,int THREADS)
{
  
  int num_subjects = int(img_info.size()/IMAGES_PER_SUBJECT);

  vector<int> start_index(num_subjects); 
  vector<int> end_index(num_subjects); 

  vector<vector<int>> avg_face;

  avg_face.resize(num_subjects,vector<int>(image_pixel[0].size()));
  avg_face_info.resize(num_subjects);
  
  string str = "avg_face_parallel/";
  // #pragma omp parallel
  // {
    // #pragma omp nowait
    #pragma omp parallel for num_threads(THREADS)
    for(int i=0;i<num_subjects;i++){

        avg_face_info[i] = img_info[IMAGES_PER_SUBJECT*i];
      
        start_index[i] = IMAGES_PER_SUBJECT * i;
        end_index[i] = IMAGES_PER_SUBJECT * (i+1) - 1 ;
      	
        #pragma omp collaspe(2) nowait
        for(int j= start_index[i];j<=end_index[i]; j++){
      
          for (int k=0;k<image_pixel[0].size();k++)
      
            avg_face[i][k] += image_pixel[j][k] ; 
      
        }
    }

    #pragma omp parallel for num_threads(THREADS)
    for(int i=0;i<avg_face.size();i++){
      
      for (int k=0;k<avg_face[0].size();k++)
        avg_face[i][k] /= IMAGES_PER_SUBJECT;

    //   print_image(avg_face[i],str+to_string(i),width,height);
    }
  
  return avg_face;
}


void initialize_centroids(vector<vector<int>>& image_pixel,vector<string> &img_info,
                          vector<vector<int>>&centroids,int THREADS)
{

    srand(10);

    //to initialize with avg face
    vector<string> avg_face_info;
    vector<vector<int>>  avg_face = parallel_average_face(image_pixel,img_info,avg_face_info,THREADS);
    
    #pragma omp parallel for num_threads(THREADS)
    for (int i = 0; i < centroids.size(); i++){
        //to initialize with avg face
        centroids[i] = avg_face[i];
        
        //random initialization
        // for(int j = 0; j < centroids[0].size(); j++){
        //     centroids[i][j] = abs(rand())%256;
        // }
    }
    
}

void cluster_assignment(vector<vector<int>>& image_pixel, vector<string> &img_info, vector<vector<int>> &centroids,
                        vector<int> &centroids_datapoint_map, int dist_criteria, int THREADS)
{

	#pragma omp parallel for num_threads(THREADS)
    for(int i = 0; i < image_pixel.size(); i++){

        double min_distance = INT_MAX*1.0,temp = 0.0;
        int closest_centroid = -1;

        for(int j = 0; j < centroids.size(); j++){

            if (dist_criteria == euc)
                temp = euclidean_distance(centroids[j], image_pixel[i], THREADS);
            else if (dist_criteria == man)
                temp = manhattan_distance(centroids[j], image_pixel[i], THREADS);
            else if (dist_criteria == che)
                temp = chebyshev_distance(centroids[j], image_pixel[i], THREADS);
            else if (dist_criteria == hel)
                temp = hellinger_distance(centroids[j], image_pixel[i], THREADS);

            if(temp<min_distance){
                min_distance = temp;
                closest_centroid = j;
            }

        }        

        centroids_datapoint_map[i] = closest_centroid;
    }

}

double move_centroid(vector<vector<int>>& image_pixel, vector<string> &img_info, vector<vector<int>> &centroids,
                    vector<int> &centroids_datapoint_map, int dist_criteria, int THREADS)
{

    double movement = 0.0;

    #pragma omp parallel for num_threads(THREADS) reduction(+:movement)
    for(int i = 0;i < centroids.size(); i++){

        vector<int> temp(centroids[0].size(),0);
        int count = 0;

        for(int j = 0; j < centroids_datapoint_map.size(); j++){

            if(centroids_datapoint_map[j] == i){

                count++;
                for(int k = 0; k < temp.size(); k++){

                    temp[k] += image_pixel[j][k];
                }
            }
        }


        if(count>0){
            

            for(int k = 0; k < temp.size(); k++){

                temp[k] /= count;
            }

            if (dist_criteria == euc)
                movement += euclidean_distance(centroids[i], temp, THREADS);
            else if (dist_criteria == man)
                movement += manhattan_distance(centroids[i], temp, THREADS);
            else if (dist_criteria == che)
                movement += chebyshev_distance(centroids[i], temp, THREADS);
            else if (dist_criteria == hel)
                movement += hellinger_distance(centroids[i], temp, THREADS);

            centroids[i] = temp;

        }

        temp.clear();
    }

    return movement;
}

void assign_centroids_label(vector<string> &img_info,vector<int> &centroids_datapoint_map,vector<string> &centroids_label,
	int THREADS){

	#pragma omp parallel for num_threads(THREADS)
    for(int i = 0; i < centroids_label.size(); i++){

        vector<int> vote(img_info.size()/IMAGES_PER_SUBJECT,0);

        for(int j = 0; j < centroids_datapoint_map.size(); j++){

            if(centroids_datapoint_map[j]==i){

                string temp = img_info[j];
                temp.erase(temp.begin());
                int index = atoi(temp.c_str()) - 1;

                vote[index] += 1;
            }
        }

        int max_vote = INT_MIN;
        string prediction = "";

        for (int j = 0; j < vote.size(); j++){

            if (vote[j] > max_vote){

                max_vote = vote[j];
                ostringstream str2;
                str2 << (j + 1);
                string sNew = str2.str();
                prediction = "S" + sNew;
            }
        }

        centroids_label[i] = prediction;
    }
}

void prediction(vector<vector<int>> &test_images,vector<string> &test_image_info,vector<vector<int>>& centroids,
                vector<string>& centroids_label,vector<string>& predicted_image_info,int dist_criteria,int THREADS)
{

    #pragma omp parallel for num_threads(THREADS)
    for(int i = 0; i < test_images.size(); i++){

        double min_distane = INT_MAX*1.0,temp = 0.0;
        int closest_centroid = -1;

        for(int j = 0; j < centroids.size(); j++){

            if (dist_criteria == euc)
                temp = euclidean_distance(centroids[j], test_images[i], THREADS);
            else if (dist_criteria == man)
                temp = manhattan_distance(centroids[j], test_images[i], THREADS);
            else if (dist_criteria == che)
                temp = chebyshev_distance(centroids[j], test_images[i], THREADS);
            else if (dist_criteria == hel)
                temp = hellinger_distance(centroids[j], test_images[i], THREADS);

            if(temp<min_distane){
                closest_centroid = j;
                min_distane = temp;
            }
        }

        predicted_image_info[i] = centroids_label[closest_centroid];
    }
}

void kmeans(vector<vector<int>> &image_pixel, vector<string> &img_info, int k, int width, int height, int dist_m,
            int THREADS,vector<string>& predicted_image_info,vector<vector<int>>& test_images,
            vector<string>& test_image_info)
{



    double movement_threshold = 0.001, movement = 1.0*INT_MAX;
    int max_iterations = 10,iterations = 0;

    vector<vector<int>> centroids(k,vector<int>(image_pixel[0].size(),0));
    vector<string> centroids_label(k," ");
    vector<int> centroids_datapoint_map(image_pixel.size(),0);

    //initialize centroids
    initialize_centroids(image_pixel,img_info,centroids,THREADS);

    while(movement > movement_threshold && iterations < max_iterations){

        //cluster assignment
        cluster_assignment(image_pixel,img_info,centroids,centroids_datapoint_map,dist_m,THREADS);

        //move centroid 
        movement = move_centroid(image_pixel,img_info,centroids,centroids_datapoint_map,dist_m,THREADS);

        cout<<"Movement: "<<movement<<"\n";
    }

    
    assign_centroids_label(img_info,centroids_datapoint_map,centroids_label,THREADS);
    
    prediction(test_images,test_image_info,centroids,centroids_label,predicted_image_info,dist_m,THREADS);
    
}

#endif