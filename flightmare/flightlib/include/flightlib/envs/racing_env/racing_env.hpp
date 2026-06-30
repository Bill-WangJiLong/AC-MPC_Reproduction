#pragma once

#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "flightlib/common/command.hpp"
#include "flightlib/common/logger.hpp"
#include "flightlib/common/quad_state.hpp"
#include "flightlib/common/types.hpp"
#include "flightlib/envs/env_base.hpp"
#include "flightlib/objects/quadrotor.hpp"

namespace flightlib {

class UnityBridge;

namespace racingenv {
enum : int {
  kNObs = 36,
  kNAct = 4,
  kNState = 13,
};
}  // namespace racingenv

struct RacingGate {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Vector<3> center{Vector<3>::Zero()};
  Vector<3> normal{Vector<3>::UnitX()};
  Vector<3> up{Vector<3>::UnitZ()};
  Vector<3> right{Vector<3>::UnitY()};
  Scalar width{1.5};
  Scalar height{1.5};
  Scalar frame_thickness{0.12};
  std::string label;

  void normalizeFrame();
  Matrix<4, 3> cornersWorld() const;
  Vector<3> worldToGate(const Ref<const Vector<3>> point) const;
  bool segmentPlaneIntersection(const Ref<const Vector<3>> p_prev,
                                const Ref<const Vector<3>> p_curr,
                                const bool direction_required,
                                Vector<3>* point) const;
  bool checkPass(const Ref<const Vector<3>> p_prev,
                 const Ref<const Vector<3>> p_curr,
                 const Scalar drone_radius,
                 const bool direction_required = true) const;
  bool checkFrameCollision(const Ref<const Vector<3>> p_prev,
                           const Ref<const Vector<3>> p_curr,
                           const Scalar drone_radius) const;
};

class RacingEnv final : public EnvBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  RacingEnv();
  RacingEnv(const std::string& cfg_path);
  ~RacingEnv();

  bool reset(Ref<Vector<>> obs, const bool random = true) override;
  Scalar step(const Ref<Vector<>> act, Ref<Vector<>> obs) override;
  bool getObs(Ref<Vector<>> obs) override;
  bool getState(Ref<Vector<>> state) override;
  bool isTerminalState(Scalar& reward) override;
  void updateExtraInfo() override;
  void addObjectsToUnity(std::shared_ptr<UnityBridge> bridge);

  bool loadParam(const YAML::Node& cfg);
  void setSeed(const int seed);

 private:
  bool loadTrack(const YAML::Node& cfg);
  bool loadTrackFromNode(const YAML::Node& track_cfg);
  void resetQuadState(const bool random);
  Command actionToCommand(const Ref<const Vector<racingenv::kNAct>> act);
  Scalar computeReward(const Ref<const Vector<3>> prev_position,
                       const Ref<const Vector<3>> curr_position,
                       const Ref<const Vector<3>> target_gate_center,
                       const Ref<const Vector<3>> body_rate,
                       const bool collision,
                       const bool gate_passed,
                       const bool race_finished) const;
  bool outOfBounds(const Ref<const Vector<3>> position) const;
  const RacingGate& currentGate() const;
  const RacingGate& futureGate(const int offset) const;
  void advanceGate();
  bool inFinishPhase() const;
  bool finishReached(const Ref<const Vector<3>> previous,
                     const Ref<const Vector<3>> current) const;
  int collisionCode(const bool frame_collision,
                    const bool ground_collision,
                    const bool out_of_bounds,
                    const bool finite_state) const;
  Vector<3> loadVector3(const YAML::Node& node, const Vector<3>& fallback) const;
  Matrix<3, 2> loadBounds(const YAML::Node& node,
                          const Matrix<3, 2>& fallback) const;

  std::shared_ptr<Quadrotor> quadrotor_ptr_;
  QuadState quad_state_;
  Command cmd_;
  Logger logger_{"RacingEnv"};

  std::vector<RacingGate> gates_;
  std::string track_name_{"horizontal"};
  Vector<3> start_position_{-2.0, 0.0, 2.0};
  Vector<3> initial_velocity_{0.0, 0.0, 0.0};
  Scalar start_yaw_{0.0};
  int current_gate_idx_{0};
  Vector<3> finish_position_{6.0, 0.0, 2.0};
  Scalar finish_radius_{0.5};
  RacingGate finish_observation_gate_;

  Matrix<3, 2> world_box_;
  Scalar drone_radius_{0.18};
  int max_episode_steps_{500};
  int steps_{0};
  bool random_reset_{true};
  Vector<3> position_noise_{0.05, 0.05, 0.02};
  Vector<3> velocity_noise_{0.02, 0.02, 0.02};
  Scalar yaw_noise_{0.02};

  Scalar thrust_max_per_motor_{8.5};
  Vector<3> omega_cmd_max_{10.0, 10.0, 4.0};

  Scalar collision_reward_{-10.0};
  Scalar gate_pass_reward_{10.0};
  Scalar finish_reward_{10.0};
  Scalar body_rate_coeff_{0.01};

  bool use_chained_gate_relative_{false};
  bool last_gate_passed_{false};
  bool last_collision_{false};
  bool last_finished_{false};
  bool last_out_of_bounds_{false};
  bool last_timeout_{false};
  int last_collision_code_{0};
  Scalar last_reward_{0.0};
  Scalar terminal_reward_{0.0};
};

}  // namespace flightlib
