/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <iomanip>

#include "particle_filter.h"

void ParticleFilter::init(double x, double y, double theta, double std[]) {

  num_particles = 200;

  particles.resize(num_particles);
  std::default_random_engine generator;
  std::normal_distribution<double> distribution(0,1);

  for(int i=0; i<num_particles; i++){

    particles[i].x = x + std[0] * distribution(generator);
    particles[i].y = y + std[1] * distribution(generator);
    particles[i].theta = theta + std[2] * distribution(generator);
    particles[i].weight = 1.;
  }

  is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {

  std::default_random_engine gen;

  for(int i=0;i<num_particles;i++){

    Particle &p = particles[i];

    double x_pred;
    double y_pred;
    double theta_pred;

    theta_pred = p.theta + yaw_rate*delta_t;
    x_pred = p.x + velocity/yaw_rate*(sin(theta_pred)-sin(p.theta));
    y_pred = p.y - velocity/yaw_rate*(cos(theta_pred)-cos(p.theta));

    std::normal_distribution<double> dist_x(x_pred,std_pos[0]);
    std::normal_distribution<double> dist_y(y_pred,std_pos[1]);
    std::normal_distribution<double> dist_theta(theta_pred,std_pos[2]);

    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);

  }

}


void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {



  for(int p_i=0;p_i<particles.size();p_i++){  // update weight for every particle

    Particle &p = particles[p_i];

    // convert observations into map coordinate system
    // rotation + translation

    //std::cout << "particle pos: " << p.x << "\t" << p.y << std::endl;
    std::vector<LandmarkObs> obs_in_map_coord;

    for(int j=0;j<observations.size();j++){

      LandmarkObs ob = observations[j]; // hold all observation within sensor range in map coord

      if(dist(observations[j].x,observations[j].y,0,0)<=sensor_range){
        // if the observations are in sensor range, change to map coordinate system


        LandmarkObs ob_in_map;
        double theta = -p.theta;
        ob_in_map.x =   ob.x * cos(theta) + ob.y * sin(theta);
        ob_in_map.y = - ob.x * sin(theta) + ob.y * cos(theta);

        ob_in_map.x += p.x;
        ob_in_map.y += p.y;

        obs_in_map_coord.push_back(ob_in_map);

      }

    }

    std::vector<LandmarkObs> map_landmarks_obs; // convert Map map_landmarks to LandMarkObs object

    for(int i=0;i<map_landmarks.landmark_list.size();i++){
      LandmarkObs map_ob;
      map_ob.x = map_landmarks.landmark_list[i].x_f;
      map_ob.y = map_landmarks.landmark_list[i].y_f;
      map_ob.id = map_landmarks.landmark_list[i].id_i;
      map_landmarks_obs.push_back(map_ob);
    }


    // calculate the weight for this particle
    double wg = 1.;
    for(int i=0;i<obs_in_map_coord.size();i++){
      // for each observations, find the closest landmark

      double min_dist = 1e100;
      int min_index = -1;

      for(int j=0;j<map_landmarks_obs.size();j++){

        double dst = dist(obs_in_map_coord[i].x,obs_in_map_coord[i].y,map_landmarks_obs[j].x,map_landmarks_obs[j].y);
        if(dst < min_dist){
          min_dist = dst;
          min_index = j;
        }
      }

      // calculate the weight from this observations and its losest landmark.
      wg *= bivariate_normal(obs_in_map_coord[i].x,
                             obs_in_map_coord[i].y,
                             map_landmarks_obs[min_index].x,
                             map_landmarks_obs[min_index].y,
                             std_landmark[0],std_landmark[1]);

    }

    p.weight = wg;
  } // end loop for particles


  // normalize weights
  double wg_sum = 0.;
  for(int i=0;i<particles.size();i++)
    wg_sum += particles[i].weight;

  for(int i=0;i<particles.size();i++)
    particles[i].weight /= wg_sum;

}

void ParticleFilter::resample() {

  std::default_random_engine gen;

  std::vector<double> weights(particles.size());
  for(int i=0;i<particles.size();i++){
    weights[i] = particles[i].weight;
  }

  std::discrete_distribution<> distribution(weights.begin(), weights.end());

  std::vector<Particle> p_new;

  for(int i=0;i<num_particles;i++){

    int weighted_index = distribution(gen);
    p_new.push_back(particles[weighted_index]);
  }

  particles = p_new;

  for(int i=0;i<particles.size();i++){
    particles[i].weight = 1.;
  }

}

void ParticleFilter::write(std::string filename) {
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
