// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once

#include "rela/thread_loop.h"
#include "rlcc/r2d2_actor.h"

class HanabiThreadLoop : public rela::ThreadLoop {
 public:
  HanabiThreadLoop(
      std::vector<std::shared_ptr<HanabiEnv>> envs,
      std::vector<std::vector<std::shared_ptr<R2D2Actor>>> actors,
      bool eval)
      : envs_(std::move(envs))
      , actors_(std::move(actors))
      , done_(envs_.size(), -1)
      , eval_(eval)
      , record_(0)
      , recordName_("zero"){
    assert(envs_.size() == actors_.size());
  }
  HanabiThreadLoop(
      std::vector<std::shared_ptr<HanabiEnv>> envs,
      std::vector<std::vector<std::shared_ptr<R2D2Actor>>> actors,
      bool eval,
      int record,
      std::string recordName)
      : envs_(std::move(envs))
      , actors_(std::move(actors))
      , done_(envs_.size(), -1)
      , eval_(eval)
      , record_(record)
      , recordName_(recordName){
    assert(envs_.size() == actors_.size());
  }

  virtual void mainLoop() override {
    if(record_==1){
      std::string log_name = "records/"+ recordName_ + ".txt";
      //std::string agent_2_main_logname = "records/"+ recordName_ + "_2m.txt";
      //std::string agent_1_partner_logname = "records/"+ recordName_ + "_1p.txt";
      //std::string agent_2_partner_logname = "records/"+ recordName_ + "_2p.txt";

      int agent_1_sim = 0;
      int agent_1_diff = 0;
      int agent_2_sim = 0;
      int agent_2_diff = 0;

      while (!terminated()) {
        // go over each envs in sequential order
        // call in seperate for-loops to maximize parallization
        for (size_t i = 0; i < envs_.size(); ++i) {
          if (done_[i] == 1) {
            continue;
          }

          auto& actors = actors_[i];

          if (envs_[i]->terminated()) {
            // we only run 1 game for evaluation
            if (eval_) {
              ++done_[i];
              if (done_[i] == 1) {
                numDone_ += 1;
                if (numDone_ == (int)envs_.size()) {
                  FILE* record_file = fopen(log_name.data(),"a");
                  fprintf(record_file, "%d\n", agent_1_sim);
                  fprintf(record_file, "%d\n", agent_1_diff);
                  fprintf(record_file, "%d\n", agent_2_sim);
                  fprintf(record_file, "%d\n", agent_2_diff);
                  fclose(record_file); 
                  return;
                }
              }
            }

            envs_[i]->reset();
            for (size_t j = 0; j < actors.size(); ++j) {
              actors[j]->reset(*envs_[i]);
            }
          }

          for (size_t j = 0; j < actors.size(); ++j) {
            actors[j]->observeBeforeAct(*envs_[i]);
          }
        }

        for (size_t i = 0; i < envs_.size(); ++i) {
          if (done_[i] == 1) {
            continue;
          }

          auto& actors = actors_[i];
          int curPlayer = envs_[i]->getCurrentPlayer();

          int main_1_act = actors[0]->recordAct(*envs_[i], curPlayer);
          int partner_1_act = actors[1]->beforeAct(curPlayer);
          int main_2_act = actors[2]->recordAct(*envs_[i], curPlayer);
          int partner_2_act = actors[3]->beforeAct(curPlayer);

          if (main_1_act != -1){
            if (main_1_act == partner_1_act){
              agent_1_sim += 1;
            } else {
              agent_1_diff +=1;
            }
          }
            //fprintf(file_1_main, "%d\n", main_1_act);
            //fprintf(file_1_partner, "%d\n", partner_1_act);
          if (main_2_act != -1){
            if (main_2_act == partner_2_act){
              agent_2_sim += 1;
            } else {
              agent_2_diff +=1;
            }
          }

        }

        for (size_t i = 0; i < envs_.size(); ++i) {
          if (done_[i] == 1) {
            continue;
          }

          auto& actors = actors_[i];
          for (size_t j = 0; j < actors.size(); ++j) {
            actors[j]->fictAct(*envs_[i]);
          }
        }

        for (size_t i = 0; i < envs_.size(); ++i) {
          if (done_[i] == 1) {
            continue;
          }

          auto& actors = actors_[i];
          for (size_t j = 0; j < actors.size(); ++j) {
            actors[j]->observeAfterAct(*envs_[i]);
          }
        }
      }   
    } else if(record_==2){
      std::string log_name_1 = "records/"+ recordName_ + "_part1.txt";
      std::string log_name_2 = "records/"+ recordName_ + "_part2.txt";

      int agent_sim_1 = 0;
      int agent_diff_1 = 0;
      int agent_sim_2 = 0;
      int agent_diff_2 = 0;

      while (!terminated()) {
        // go over each envs in sequential order
        // call in seperate for-loops to maximize parallization
        for (size_t i = 0; i < envs_.size(); ++i) {
          if (done_[i] == 1) {
            continue;
          }

          auto& actors = actors_[i];

          if (envs_[i]->terminated()) {
            // we only run 1 game for evaluation
            if (eval_) {
              ++done_[i];
              if (done_[i] == 1) {
                numDone_ += 1;
                if (numDone_ == (int)envs_.size()) {
                  FILE* record_file_1 = fopen(log_name_1.data(),"a");
                  fprintf(record_file_1, "%d\n", agent_sim_1);
                  fprintf(record_file_1, "%d\n", agent_diff_1);
                  fclose(record_file_1);      
                  FILE* record_file_2 = fopen(log_name_2.data(),"a");
                  fprintf(record_file_2, "%d\n", agent_sim_2);
                  fprintf(record_file_2, "%d\n", agent_diff_2);
                  fclose(record_file_2);    
                  return;
                }
              }
            }

            envs_[i]->reset();
            for (size_t j = 0; j < actors.size(); ++j) {
              actors[j]->reset(*envs_[i]);
            }
          }

          for (size_t j = 0; j < actors.size(); ++j) {
            actors[j]->observeBeforeAct(*envs_[i]);
          }
        }

        for (size_t i = 0; i < envs_.size(); ++i) {
          if (done_[i] == 1) {
            continue;
          }

          auto& actors = actors_[i];
          int curPlayer = envs_[i]->getCurrentPlayer();

          actors[0]->act(*envs_[i], curPlayer);
          int main_1_act = actors[1]->recordAct(*envs_[i], curPlayer);
          int partner_1_act = actors[2]->beforeAct(curPlayer);
          int main_2_act = actors[3]->recordAct(*envs_[i], curPlayer);
          int partner_2_act = actors[4]->beforeAct(curPlayer);

          if (main_1_act != -1){
            if (main_1_act == partner_1_act){
              agent_sim_1 += 1;
            } else {
              agent_diff_1 +=1;
            }
          }

          if (main_2_act != -1){
            if (main_2_act == partner_2_act){
              agent_sim_2 += 1;
            } else {
              agent_diff_2 +=1;
            }
          }

        }

        for (size_t i = 0; i < envs_.size(); ++i) {
          if (done_[i] == 1) {
            continue;
          }

          auto& actors = actors_[i];
          for (size_t j = 0; j < actors.size(); ++j) {
            actors[j]->fictAct(*envs_[i]);
          }
        }

        for (size_t i = 0; i < envs_.size(); ++i) {
          if (done_[i] == 1) {
            continue;
          }

          auto& actors = actors_[i];
          for (size_t j = 0; j < actors.size(); ++j) {
            actors[j]->observeAfterAct(*envs_[i]);
          }
        }
      }


    } else {
      while (!terminated()) {
        // go over each envs in sequential order
        // call in seperate for-loops to maximize parallization
        for (size_t i = 0; i < envs_.size(); ++i) {
          if (done_[i] == 1) {
            continue;
          }

          auto& actors = actors_[i];

          if (envs_[i]->terminated()) {
            // we only run 1 game for evaluation
            if (eval_) {
              ++done_[i];
              if (done_[i] == 1) {
                numDone_ += 1;
                if (numDone_ == (int)envs_.size()) {
                  return;
                }
              }
            }

            envs_[i]->reset();
            for (size_t j = 0; j < actors.size(); ++j) {
              actors[j]->reset(*envs_[i]);
            }
          }

          for (size_t j = 0; j < actors.size(); ++j) {
            actors[j]->observeBeforeAct(*envs_[i]);
          }
        }

        for (size_t i = 0; i < envs_.size(); ++i) {
          if (done_[i] == 1) {
            continue;
          }

          auto& actors = actors_[i];
          int curPlayer = envs_[i]->getCurrentPlayer();
          for (size_t j = 0; j < actors.size(); ++j) {
            actors[j]->act(*envs_[i], curPlayer);
          }
        }

        for (size_t i = 0; i < envs_.size(); ++i) {
          if (done_[i] == 1) {
            continue;
          }

          auto& actors = actors_[i];
          for (size_t j = 0; j < actors.size(); ++j) {
            actors[j]->fictAct(*envs_[i]);
          }
        }

        for (size_t i = 0; i < envs_.size(); ++i) {
          if (done_[i] == 1) {
            continue;
          }

          auto& actors = actors_[i];
          for (size_t j = 0; j < actors.size(); ++j) {
            actors[j]->observeAfterAct(*envs_[i]);
          }
        }
      }
    }


  }

 private:
  std::vector<std::shared_ptr<HanabiEnv>> envs_;
  std::vector<std::vector<std::shared_ptr<R2D2Actor>>> actors_;
  std::vector<int8_t> done_;
  const bool eval_;
  const int record_;
  std::string recordName_;
  int numDone_ = 0;
};
