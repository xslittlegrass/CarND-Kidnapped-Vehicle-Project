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
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  num_particles = 200;

  particles.resize(num_particles);
  std::default_random_engine gen;

  for(int i=0; i<num_particles; i++){

    std::normal_distribution<double> dist_x(x,std[0]);
    std::normal_distribution<double> dist_y(y,std[1]);
    std::normal_distribution<double> dist_theta(theta,std[2]);

    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
    particles[i].weight = 1.;

  }

  is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/


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

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

  for(int i=0;i<observations.size();i++){

    double min_distance=dist(predicted[0].x,predicted[0].y,observations[i].x,observations[i].y);

    for(int j=0;j<predicted.size();j++){
      double current_dist = dist(predicted[j].x,predicted[j].y,observations[i].x,observations[i].y);
      if(current_dist < min_distance){
        min_distance = current_dist;
        observations[i].id = predicted[j].id;
      }
    }
  }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html



  for(int i=0;i<particles.size();i++){  // update weight for every particle

    Particle &p = particles[i];

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


    dataAssociation(map_landmarks_obs, obs_in_map_coord); // associate the map landmark to each measured landmark

    double wg = 1.;

    for(int i=0;i<obs_in_map_coord.size();i++){


      double ob_x = obs_in_map_coord[i].x;
      double ob_y = obs_in_map_coord[i].y;
      int map_id = obs_in_map_coord[i].id;

      // find the map landmark that is associated with the ith observations
      LandmarkObs map_lm;
      for(int k=0;k<map_landmarks_obs.size();k++){
        if(map_landmarks_obs[k].id == map_id){
          map_lm = map_landmarks_obs[k];
          break;
        }
      }

      double landmark_x = map_lm.x;
      double landmark_y = map_lm.y;

      double sigma_x = std_landmark[0];
      double sigma_y = std_landmark[1];

      double expfactor = (ob_x-landmark_x)*(ob_x-landmark_x)/2./sigma_x/sigma_x;
      expfactor +=  (ob_y-landmark_y)*(ob_y-landmark_y)/2./sigma_y/sigma_y;
      expfactor = - expfactor;

      wg *= 1./(2*M_PI*sigma_x*sigma_y) * exp(expfactor);

      //      std::cout << "expfactor \t" << expfactor << std::endl;
    }

    p.weight = wg;
  } // end loop for particles

  // normalize weights

  double wg_sum = 0.;
  for(int i=0;i<particles.size();i++)
    wg_sum += particles[i].weight;

  for(int i=0;i<particles.size();i++)
    particles[i].weight /= wg_sum;

  //for(int i=0;i<particles.size();i++)
    //std::cout << "particle weight: " << std::setprecision(6) << particles[i].weight << std::endl;
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

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
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
