# GNC Toolkit

## TODO

### Numerical Integrators
- [ ] Runge-Kutta 4 (RK4, fixed step)
- [ ] Runge-Kutta-Fehlberg (RK45, variable step)
- [ ] Dormand-Prince (RK853, variable step)

### Propagators
- [ ] Cowell method (Integrate eqs of motion with disturbances)
- [ ] Fix kepler propagator

### Attitude Dynamics
- [ ] Rigid body euler equations

### Deterministic Attitude Determination
- [ ] TRIAD algorithm (using 2 vector e.g. sun and mag)
- [ ] QUEST algorithm (using multiple vectors)

### Classical Control Algorithms
- [ ] PID control (add anti-windup)
- [ ] B-dot controller (magnetic detumbling)

### Optimal Control Algorithms
- [ ] LQR controller (infinite horizon, via solving ARE)
- [ ] LQE controller (duality of LQR for estimation)
- [ ] Sliding mode controller
- [ ] MPC
- [ ] Feedback Linearization (cancel out non-linear terms)

### C++ Library
- [ ] Develop cpp library further

### Guidance & Mission Analysis Tools
- [ ] Orbital Manouvers
    - [ ] Hohmann Transfer
    - [ ] Bi-Elliptic Transfer
    - [ ] Phasing Manouvers
    - [ ] Plane change manouvers
- [ ] Intercept & Rendezvous
    - [ ] Lambert Problem
    - [ ] CW eqs (Hill's equations, linearized relative motion)

### Actuator & Sensor Models
- [ ] Sensors
    - [ ] Star Tracker
    - [ ] Sun Sensor
    - [ ] Magnetometer
    - [ ] Gyroscope
- [ ] Actuators
    - [ ] RWs
    - [ ] MTQs
    - [ ] Thrusters (PWM logic for on/off thrusters)
