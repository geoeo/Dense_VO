import numpy as np
import math


class GradientStepManager:
    def __init__(self, alpha_start, alpha_min, alpha_step,alpha_change_rate, gradient_monitoring_window_start, gradient_monitoring_window_size):
        self.current_alpha = alpha_start
        self.alpha_min = alpha_min
        self.alpha_step = alpha_step
        self.gradient_monitoring_window_start = gradient_monitoring_window_start
        self.gradient_monitoring_window_size = gradient_monitoring_window_size
        self.gradient_monitoring_window = np.full((1,gradient_monitoring_window_size), False)
        self.last_error_mean_abs = -1000
        self.alpha_change_rate = alpha_change_rate

    #TODO track continuously
    def track_gradient(self, current_error_mean_abs, current_iteration):
        if self.gradient_monitoring_window_start < current_iteration < self.gradient_monitoring_window_start + self.gradient_monitoring_window_size:
            self.gradient_monitoring_window[0, current_iteration - self.gradient_monitoring_window_start] = current_error_mean_abs >= self.last_error_mean_abs

    def save_previous_mean_error(self, current_error_mean_abs, current_iteration):
        if current_iteration > self.gradient_monitoring_window_start:
            self.last_error_mean_abs = current_error_mean_abs

    def analyze_gradient_history(self, current_iteration):

        new_alpha = self.current_alpha
        if current_iteration == self.gradient_monitoring_window_size and current_iteration > 0:
            number_of_error_increases = np.sum(self.gradient_monitoring_window[0])
            if number_of_error_increases > math.floor(self.gradient_monitoring_window_size/2):
                print('switching alpha!')
                new_alpha *= -1
                self.alpha_step *= -1
                self.alpha_min *= -1

        if self.alpha_change_rate > 0:
            if current_iteration > 0 and current_iteration % self.alpha_change_rate == 0:
                if math.fabs(new_alpha) > math.fabs(self.alpha_min):
                    new_alpha -= self.alpha_step
                    print('new alpha: ', new_alpha)

        self.current_alpha = new_alpha
