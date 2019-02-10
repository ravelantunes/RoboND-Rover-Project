import numpy as np


# This is where you can build a decision tree for determining throttle, brake and steer 
# commands based on the output of the perception_step() function
def decision_step(Rover):
  
  # If no angles set. do nothing
  if Rover.nav_angles is None:
    print("No angles sent to Rover. Doing nothing.")
    Rover.throttle = Rover.throttle_set
    Rover.steer = 0
    Rover.brake = 0
    return Rover

    
  if Rover.mode == 'stuck':
    time_spent_stuck = Rover.total_time - Rover.last_time_stuck
    print("time stuck", time_spent_stuck)
    if time_spent_stuck > 3:
      Rover.mode = 'forward'
    else:
      Rover.throttle = -0.2          
      Rover.steer = 0
      Rover.brake = 0

  # Check for Rover.mode status
  if Rover.mode == 'forward': 
      # Check the extent of navigable terrain
      if len(Rover.nav_angles) >= Rover.stop_forward:  
          # If mode is forward, navigable terrain looks good 
          # and velocity is below max, then throttle 
          if Rover.vel < Rover.max_vel:
              # Set throttle value to throttle setting
              Rover.throttle = Rover.throttle_set
              
              if Rover.vel < 0.1 and Rover.throttle == Rover.throttle_set:

                # Make it wasn't just a recent stuck
                if Rover.last_time_stuck is None or (Rover.total_time - Rover.last_time_stuck) > 5:
                  Rover.mode = 'stuck'
                  Rover.last_time_stuck = Rover.total_time

          else: # Else coast
              Rover.throttle = 0
          Rover.brake = 0
          # Set steering to average angle clipped to the range +/- 15
          steer_angle = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15)
          # print("Steer angle: ", steer_angle)
          Rover.steer = steer_angle
      # If there's a lack of navigable terrain pixels then go to 'stop' mode
      elif len(Rover.nav_angles) < Rover.stop_forward:
              # Set mode to "stop" and hit the brakes!
              Rover.throttle = 0
              # Set brake to stored brake value
              Rover.brake = Rover.brake_set
              Rover.steer = 0
              Rover.mode = 'stop'

  # If we're already in "stop" mode then make different decisions
  elif Rover.mode == 'stop':
      # If we're in stop mode but still moving keep braking
      if Rover.vel > 0.2:
          Rover.throttle = 0
          Rover.brake = Rover.brake_set
          Rover.steer = 0
      # If we're not moving (vel < 0.2) then do something else
      elif Rover.vel <= 0.2:
          # Now we're stopped and we have vision data to see if there's a path forward
          if len(Rover.nav_angles) < Rover.go_forward:
              Rover.throttle = 0
              # Release the brake to allow turning
              Rover.brake = 0
              # Turn range is +/- 15 degrees, when stopped the next line will induce 4-wheel turning
              Rover.steer = -15 # Could be more clever here about which way to turn
          # If we're stopped but see sufficient navigable terrain in front then go!
          if len(Rover.nav_angles) >= Rover.go_forward:
              # Set throttle back to stored value
              Rover.throttle = Rover.throttle_set
              # Release the brake
              Rover.brake = 0
              # Set steer to mean angle
              Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15)
              Rover.mode = 'forward'
        
  # If in a state where want to pickup a rock send pickup command
  if Rover.near_sample and Rover.vel == 0 and not Rover.picking_up:
      Rover.send_pickup = True
      
  return Rover

