#include "flightlib/envs/racing_env/racing_env.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>

#include <eigen3/Eigen/Geometry>

#ifdef FLIGHTLIB_BUILD_UNITY_BRIDGE
#include "flightlib/bridges/unity_bridge.hpp"
#endif
#include "flightlib/dynamics/quadrotor_dynamics.hpp"

namespace flightlib {

namespace {

Vector<3> normalizedOrFallback(const Vector<3>& v, const Vector<3>& fallback) {
  const Scalar n = v.norm();
  if (!std::isfinite(n) || n < 1e-6) return fallback;
  return v / n;
}

Scalar clampScalar(const Scalar value, const Scalar low, const Scalar high) {
  return std::min(std::max(value, low), high);
}

std::string defaultRacingCfgPath() {
  const char* flightmare_path = std::getenv("FLIGHTMARE_PATH");
  if (flightmare_path != nullptr) {
    return std::string(flightmare_path) + "/flightlib/configs/racing_env.yaml";
  }
  return "flightlib/configs/racing_env.yaml";
}

}  // namespace

void RacingGate::normalizeFrame() {
  normal = normalizedOrFallback(normal, Vector<3>::UnitX());
  up = up - up.dot(normal) * normal;
  up = normalizedOrFallback(up, Vector<3>::UnitZ());
  right = normalizedOrFallback(up.cross(normal), Vector<3>::UnitY());
}

Matrix<4, 3> RacingGate::cornersWorld() const {
  const Scalar half_w = 0.5 * width;
  const Scalar half_h = 0.5 * height;
  Matrix<4, 3> corners;
  corners.row(0) = center - half_w * right - half_h * up;
  corners.row(1) = center + half_w * right - half_h * up;
  corners.row(2) = center + half_w * right + half_h * up;
  corners.row(3) = center - half_w * right + half_h * up;
  return corners;
}

Vector<3> RacingGate::worldToGate(const Ref<const Vector<3>> point) const {
  const Vector<3> delta = point - center;
  return (Vector<3>() << delta.dot(right), delta.dot(up), delta.dot(normal))
    .finished();
}

bool RacingGate::segmentPlaneIntersection(const Ref<const Vector<3>> p_prev,
                                          const Ref<const Vector<3>> p_curr,
                                          const bool direction_required,
                                          Vector<3>* point) const {
  const Scalar d0 = (p_prev - center).dot(normal);
  const Scalar d1 = (p_curr - center).dot(normal);

  if (direction_required) {
    if (!(d0 < 0.0 && d1 >= 0.0)) return false;
  } else {
    if (d0 == 0.0 && d1 == 0.0) return false;
    if (d0 * d1 > 0.0) return false;
  }

  const Scalar denom = d1 - d0;
  if (std::abs(denom) < 1e-6) return false;

  const Scalar t = clampScalar(-d0 / denom, 0.0, 1.0);
  *point = p_prev + t * (p_curr - p_prev);
  return true;
}

bool RacingGate::checkPass(const Ref<const Vector<3>> p_prev,
                           const Ref<const Vector<3>> p_curr,
                           const Scalar drone_radius,
                           const bool direction_required) const {
  Vector<3> point;
  if (!segmentPlaneIntersection(p_prev, p_curr, direction_required, &point))
    return false;

  const Vector<3> local = worldToGate(point);
  const Scalar half_w = std::max<Scalar>(0.0, 0.5 * width - drone_radius);
  const Scalar half_h = std::max<Scalar>(0.0, 0.5 * height - drone_radius);
  return std::abs(local.x()) <= half_w && std::abs(local.y()) <= half_h;
}

bool RacingGate::checkFrameCollision(const Ref<const Vector<3>> p_prev,
                                     const Ref<const Vector<3>> p_curr,
                                     const Scalar drone_radius) const {
  Vector<3> point;
  if (!segmentPlaneIntersection(p_prev, p_curr, false, &point)) return false;

  const Vector<3> local = worldToGate(point);
  const Scalar inner_w = std::max<Scalar>(0.0, 0.5 * width - drone_radius);
  const Scalar inner_h = std::max<Scalar>(0.0, 0.5 * height - drone_radius);
  const Scalar outer_w = 0.5 * width + frame_thickness + drone_radius;
  const Scalar outer_h = 0.5 * height + frame_thickness + drone_radius;

  const bool inside_inner =
    std::abs(local.x()) <= inner_w && std::abs(local.y()) <= inner_h;
  const bool inside_outer =
    std::abs(local.x()) <= outer_w && std::abs(local.y()) <= outer_h;
  return inside_outer && !inside_inner;
}

RacingEnv::RacingEnv() : RacingEnv(defaultRacingCfgPath()) {}

RacingEnv::RacingEnv(const std::string& cfg_path)
  : EnvBase(),
    quadrotor_ptr_(std::make_shared<Quadrotor>()),
    world_box_((Matrix<3, 2>() << -8.0, 12.0, -6.0, 6.0, 0.0, 7.0)
                 .finished()) {
  YAML::Node cfg = YAML::LoadFile(cfg_path);

  QuadrotorDynamics dynamics;
  dynamics.updateParams(cfg);
  quadrotor_ptr_->updateDynamics(dynamics);

  Matrix<3, 2> safety_box;
  safety_box << -1e6, 1e6, -1e6, 1e6, -1e6, 1e6;
  quadrotor_ptr_->setWorldBox(safety_box);

  obs_dim_ = racingenv::kNObs;
  act_dim_ = racingenv::kNAct;
  state_dim_ = racingenv::kNState;

  loadParam(cfg);
  resetQuadState(false);
  updateExtraInfo();
}

RacingEnv::~RacingEnv() {}

bool RacingEnv::loadParam(const YAML::Node& cfg) {
  if (cfg["racing_env"]) {
    const YAML::Node env_cfg = cfg["racing_env"];
    if (env_cfg["sim_dt"]) sim_dt_ = env_cfg["sim_dt"].as<Scalar>();
    if (env_cfg["max_t"]) max_t_ = env_cfg["max_t"].as<Scalar>();
    if (env_cfg["max_episode_steps"])
      max_episode_steps_ = env_cfg["max_episode_steps"].as<int>();
    if (env_cfg["drone_radius"])
      drone_radius_ = env_cfg["drone_radius"].as<Scalar>();
    if (env_cfg["random_reset"])
      random_reset_ = env_cfg["random_reset"].as<bool>();
    if (env_cfg["world_box"])
      world_box_ = loadBounds(env_cfg["world_box"], world_box_);
  }

  if (cfg["initial_state"]) {
    const YAML::Node init_cfg = cfg["initial_state"];
    if (init_cfg["position"])
      start_position_ = loadVector3(init_cfg["position"], start_position_);
    if (init_cfg["velocity"])
      initial_velocity_ = loadVector3(init_cfg["velocity"], initial_velocity_);
    if (init_cfg["yaw"]) start_yaw_ = init_cfg["yaw"].as<Scalar>();
    if (init_cfg["position_noise"])
      position_noise_ = loadVector3(init_cfg["position_noise"], position_noise_);
    if (init_cfg["velocity_noise"])
      velocity_noise_ = loadVector3(init_cfg["velocity_noise"], velocity_noise_);
    if (init_cfg["yaw_noise"]) yaw_noise_ = init_cfg["yaw_noise"].as<Scalar>();
  }

  if (cfg["action"]) {
    const YAML::Node action_cfg = cfg["action"];
    if (action_cfg["thrust_max_per_motor"])
      thrust_max_per_motor_ = action_cfg["thrust_max_per_motor"].as<Scalar>();
    if (action_cfg["omega_max"])
      omega_cmd_max_ = loadVector3(action_cfg["omega_max"], omega_cmd_max_);
  }

  if (cfg["reward"]) {
    const YAML::Node reward_cfg = cfg["reward"];
    if (reward_cfg["collision_reward"])
      collision_reward_ = reward_cfg["collision_reward"].as<Scalar>();
    if (reward_cfg["gate_pass_reward"])
      gate_pass_reward_ = reward_cfg["gate_pass_reward"].as<Scalar>();
    if (reward_cfg["finish_reward"])
      finish_reward_ = reward_cfg["finish_reward"].as<Scalar>();
    if (reward_cfg["body_rate_coeff"])
      body_rate_coeff_ = reward_cfg["body_rate_coeff"].as<Scalar>();
  }

  if (cfg["observation"] && cfg["observation"]["track_obs_mode"]) {
    const std::string mode = cfg["observation"]["track_obs_mode"].as<std::string>();
    use_chained_gate_relative_ = mode == "chained_gate_relative";
  }

  return loadTrack(cfg);
}

bool RacingEnv::loadTrack(const YAML::Node& cfg) {
  if (!cfg["track"]) return false;
  const YAML::Node track_cfg = cfg["track"];

  if (track_cfg["track_path"]) {
    YAML::Node loaded = YAML::LoadFile(track_cfg["track_path"].as<std::string>());
    if (loaded["track"]) loaded = loaded["track"];
    return loadTrackFromNode(loaded);
  }
  return loadTrackFromNode(track_cfg);
}

bool RacingEnv::loadTrackFromNode(const YAML::Node& track_cfg) {
  if (!track_cfg || !track_cfg["gates"] || !track_cfg["gates"].IsSequence())
    return false;

  if (track_cfg["name"]) track_name_ = track_cfg["name"].as<std::string>();
  if (track_cfg["start"]) {
    const YAML::Node start_cfg = track_cfg["start"];
    if (start_cfg["position"])
      start_position_ = loadVector3(start_cfg["position"], start_position_);
    if (start_cfg["yaw"]) start_yaw_ = start_cfg["yaw"].as<Scalar>();
  }
  if (track_cfg["world_bounds"])
    world_box_ = loadBounds(track_cfg["world_bounds"], world_box_);

  gates_.clear();
  for (const YAML::Node& gate_cfg : track_cfg["gates"]) {
    RacingGate gate;
    gate.center = loadVector3(gate_cfg["center"], gate.center);
    gate.normal = loadVector3(gate_cfg["normal"], gate.normal);
    gate.up = loadVector3(gate_cfg["up"], gate.up);
    if (gate_cfg["width"]) gate.width = gate_cfg["width"].as<Scalar>();
    if (gate_cfg["height"]) gate.height = gate_cfg["height"].as<Scalar>();
    if (gate_cfg["frame_thickness"])
      gate.frame_thickness = gate_cfg["frame_thickness"].as<Scalar>();
    if (gate_cfg["label"]) gate.label = gate_cfg["label"].as<std::string>();
    gate.normalizeFrame();
    gates_.push_back(gate);
  }
  if (gates_.empty() || !track_cfg["finish"]) return false;

  const YAML::Node finish_cfg = track_cfg["finish"];
  if (!finish_cfg["position"] || !finish_cfg["radius"]) return false;
  finish_position_ = loadVector3(finish_cfg["position"], finish_position_);
  finish_radius_ = finish_cfg["radius"].as<Scalar>();
  if (!finish_position_.allFinite() || !std::isfinite(finish_radius_) ||
      finish_radius_ <= 0.0)
    return false;

  finish_observation_gate_ = gates_.back();
  finish_observation_gate_.center = finish_position_;
  finish_observation_gate_.width = 2.0 * finish_radius_;
  finish_observation_gate_.height = 2.0 * finish_radius_;
  finish_observation_gate_.frame_thickness = 0.0;
  finish_observation_gate_.label = "FINISH";
  finish_observation_gate_.normalizeFrame();
  return true;
}

bool RacingEnv::reset(Ref<Vector<>> obs, const bool random) {
  current_gate_idx_ = 0;
  steps_ = 0;
  last_gate_passed_ = false;
  last_collision_ = false;
  last_finished_ = false;
  last_out_of_bounds_ = false;
  last_timeout_ = false;
  last_collision_code_ = 0;
  last_reward_ = 0.0;
  terminal_reward_ = 0.0;

  resetQuadState(random && random_reset_);
  return getObs(obs);
}

void RacingEnv::resetQuadState(const bool random) {
  quad_state_.setZero();
  quad_state_.p = start_position_;
  quad_state_.v = initial_velocity_;
  Scalar yaw = start_yaw_;

  if (random) {
    for (int i = 0; i < 3; i++) {
      quad_state_.p(i) += position_noise_(i) * uniform_dist_(random_gen_);
      quad_state_.v(i) += velocity_noise_(i) * uniform_dist_(random_gen_);
    }
    yaw += yaw_noise_ * uniform_dist_(random_gen_);
  }

  const Eigen::AngleAxis<Scalar> yaw_angle(yaw, Vector<3>::UnitZ());
  const Quaternion q(yaw_angle);
  quad_state_.q(q);
  quad_state_.w.setZero();
  quadrotor_ptr_->reset(quad_state_);
  cmd_ = Command(0.0, -Gz, Vector<3>::Zero());
}

Scalar RacingEnv::step(const Ref<Vector<>> act, Ref<Vector<>> obs) {
  quadrotor_ptr_->getState(&quad_state_);
  const Vector<3> prev_position = quad_state_.p;
  const bool was_finish_phase = inFinishPhase();
  const RacingGate* target_gate = was_finish_phase ? nullptr : &currentGate();
  const Vector<3> target_position =
    was_finish_phase ? finish_position_ : target_gate->center;

  cmd_ = actionToCommand(act.segment<racingenv::kNAct>(0));
  quadrotor_ptr_->run(cmd_, sim_dt_);
  quadrotor_ptr_->getState(&quad_state_);
  steps_ += 1;

  const Vector<3> curr_position = quad_state_.p;
  last_gate_passed_ =
    target_gate != nullptr &&
    target_gate->checkPass(prev_position, curr_position, drone_radius_, true);
  const bool frame_collision =
    target_gate != nullptr &&
    target_gate->checkFrameCollision(prev_position, curr_position, drone_radius_);

  if (last_gate_passed_) advanceGate();
  last_finished_ = finishReached(prev_position, curr_position);

  last_out_of_bounds_ = outOfBounds(curr_position);
  const bool ground_collision =
    curr_position.z() <= world_box_(2, 0) + drone_radius_;
  const bool finite_state = quad_state_.valid() && quad_state_.qx.norm() > 1e-6;
  last_timeout_ = steps_ >= max_episode_steps_;
  last_collision_ =
    frame_collision || ground_collision || last_out_of_bounds_ || !finite_state;
  last_collision_code_ =
    collisionCode(frame_collision, ground_collision, last_out_of_bounds_,
                  finite_state);

  last_reward_ =
    computeReward(prev_position, curr_position, target_position, quad_state_.w,
                  last_collision_, last_gate_passed_, last_finished_);
  getObs(obs);
  return last_reward_;
}

Command RacingEnv::actionToCommand(
  const Ref<const Vector<racingenv::kNAct>> act) {
  Vector<racingenv::kNAct> clipped =
    act.cwiseMax(-1.0).cwiseMin(1.0);

  const Scalar mass = quadrotor_ptr_->getMass();
  const Scalar force_mean = (thrust_max_per_motor_ * 4.0 / mass) / 2.0;
  const Scalar force_std = force_mean;
  const Scalar mass_normalized_thrust =
    clipped(0) * force_std + force_mean;
  const Vector<3> omega_cmd = clipped.segment<3>(1).cwiseProduct(omega_cmd_max_);

  return Command(cmd_.t + sim_dt_, mass_normalized_thrust, omega_cmd);
}

bool RacingEnv::getObs(Ref<Vector<>> obs) {
  quadrotor_ptr_->getState(&quad_state_);
  if (obs.size() != racingenv::kNObs) return false;

  obs.setZero();
  obs.segment<3>(0) = quad_state_.v;

  Quaternion q = quad_state_.q();
  q.normalize();
  const Matrix<3, 3> rot = q.toRotationMatrix();
  for (int r = 0; r < 3; r++) {
    for (int c = 0; c < 3; c++) {
      obs(3 + r * 3 + c) = rot(r, c);
    }
  }

  const Matrix<4, 3> gate1_corners = futureGate(0).cornersWorld();
  const Matrix<4, 3> gate2_corners = futureGate(1).cornersWorld();
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 3; j++) {
      obs(12 + i * 3 + j) = gate1_corners(i, j) - quad_state_.p(j);
      obs(24 + i * 3 + j) =
        use_chained_gate_relative_
          ? gate2_corners(i, j) - gate1_corners(i, j)
          : gate2_corners(i, j) - quad_state_.p(j);
    }
  }
  return obs.allFinite();
}

bool RacingEnv::getState(Ref<Vector<>> state) {
  quadrotor_ptr_->getState(&quad_state_);
  if (state.size() != racingenv::kNState) return false;
  state.segment<3>(0) = quad_state_.p;
  state.segment<4>(3) = quad_state_.qx;
  state.segment<3>(7) = quad_state_.v;
  state.segment<3>(10) = quad_state_.w;
  return state.allFinite();
}

Scalar RacingEnv::computeReward(const Ref<const Vector<3>> prev_position,
                                const Ref<const Vector<3>> curr_position,
                                const Ref<const Vector<3>> target_gate_center,
                                const Ref<const Vector<3>> body_rate,
                                const bool collision,
                                const bool gate_passed,
                                const bool race_finished) const {
  if (collision) return collision_reward_;
  if (race_finished)
    return finish_reward_ + (gate_passed ? gate_pass_reward_ : 0.0);
  if (gate_passed) return gate_pass_reward_;

  const Scalar prev_distance = (target_gate_center - prev_position).norm();
  const Scalar curr_distance = (target_gate_center - curr_position).norm();
  return prev_distance - curr_distance - body_rate_coeff_ * body_rate.norm();
}

bool RacingEnv::isTerminalState(Scalar& reward) {
  reward = terminal_reward_;
  return last_collision_ || last_finished_ || last_timeout_;
}

void RacingEnv::updateExtraInfo() {
  extra_info_["gate_index"] = static_cast<float>(current_gate_idx_);
  extra_info_["gate_passed"] = last_gate_passed_ ? 1.0f : 0.0f;
  extra_info_["finish_phase"] = inFinishPhase() ? 1.0f : 0.0f;
  extra_info_["finish_distance"] = (quad_state_.p - finish_position_).norm();
  extra_info_["collision"] = last_collision_ ? 1.0f : 0.0f;
  extra_info_["collision_code"] = static_cast<float>(last_collision_code_);
  extra_info_["finished"] = last_finished_ ? 1.0f : 0.0f;
  extra_info_["out_of_bounds"] = last_out_of_bounds_ ? 1.0f : 0.0f;
  extra_info_["timeout"] = last_timeout_ ? 1.0f : 0.0f;
  extra_info_["speed"] = quad_state_.v.norm();
  extra_info_["x"] = quad_state_.p.x();
  extra_info_["y"] = quad_state_.p.y();
  extra_info_["z"] = quad_state_.p.z();
}

void RacingEnv::addObjectsToUnity(std::shared_ptr<UnityBridge> bridge) {
#ifdef FLIGHTLIB_BUILD_UNITY_BRIDGE
  bridge->addQuadrotor(quadrotor_ptr_);
#else
  (void)bridge;
#endif
}

void RacingEnv::setSeed(const int seed) {
  std::srand(seed);
  random_gen_.seed(seed);
}

bool RacingEnv::outOfBounds(const Ref<const Vector<3>> position) const {
  for (int i = 0; i < 3; i++) {
    if (position(i) < world_box_(i, 0) || position(i) > world_box_(i, 1))
      return true;
  }
  return false;
}

const RacingGate& RacingEnv::currentGate() const {
  return gates_[std::min<int>(current_gate_idx_, gates_.size() - 1)];
}

const RacingGate& RacingEnv::futureGate(const int offset) const {
  const int index = current_gate_idx_ + offset;
  return index < static_cast<int>(gates_.size()) ? gates_[index]
                                                 : finish_observation_gate_;
}

void RacingEnv::advanceGate() {
  if (current_gate_idx_ < static_cast<int>(gates_.size())) current_gate_idx_++;
}

bool RacingEnv::inFinishPhase() const {
  return current_gate_idx_ >= static_cast<int>(gates_.size());
}

bool RacingEnv::finishReached(const Ref<const Vector<3>> previous,
                              const Ref<const Vector<3>> current) const {
  if (!inFinishPhase()) return false;
  const Vector<3> segment = current - previous;
  const Scalar squared_length = segment.squaredNorm();
  if (squared_length <= 1e-8)
    return (current - finish_position_).norm() <= finish_radius_;
  const Scalar interpolation = clampScalar(
    (finish_position_ - previous).dot(segment) / squared_length, 0.0, 1.0);
  const Vector<3> closest = previous + interpolation * segment;
  return (closest - finish_position_).norm() <= finish_radius_;
}

int RacingEnv::collisionCode(const bool frame_collision,
                             const bool ground_collision,
                             const bool out_of_bounds,
                             const bool finite_state) const {
  if (!finite_state) return 4;
  if (out_of_bounds) return 1;
  if (ground_collision) return 2;
  if (frame_collision) return 3;
  return 0;
}

Vector<3> RacingEnv::loadVector3(const YAML::Node& node,
                                 const Vector<3>& fallback) const {
  if (!node || !node.IsSequence() || node.size() < 3) return fallback;
  return (Vector<3>() << node[0].as<Scalar>(), node[1].as<Scalar>(),
          node[2].as<Scalar>())
    .finished();
}

Matrix<3, 2> RacingEnv::loadBounds(const YAML::Node& node,
                                   const Matrix<3, 2>& fallback) const {
  if (!node || !node.IsSequence() || node.size() < 3) return fallback;
  Matrix<3, 2> bounds = fallback;
  for (int i = 0; i < 3; i++) {
    if (!node[i].IsSequence() || node[i].size() < 2) return fallback;
    bounds(i, 0) = node[i][0].as<Scalar>();
    bounds(i, 1) = node[i][1].as<Scalar>();
  }
  return bounds;
}

}  // namespace flightlib
