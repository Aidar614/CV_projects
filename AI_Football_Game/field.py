class Field:
    def __init__(self, length=120, width=80,
                 penalty_box_length=17.8, penalty_box_width=43,
                 goal_box_length=5.5, goal_box_width=19,
                 penalty_spot_distance=12, centre_circle_radius=10):
        self.length = length
        self.width = width
        self.penalty_box_length = penalty_box_length
        self.penalty_box_width = penalty_box_width
        self.goal_box_length = goal_box_length
        self.goal_box_width = goal_box_width
        self.penalty_spot_distance = penalty_spot_distance
        self.centre_circle_radius = centre_circle_radius

    def vertices(self):
      return [
          (0, 0),  # 1
          (0, (self.width - self.penalty_box_width) / 2),  # 2
          (0, (self.width - self.goal_box_width) / 2),  # 3
          (0, (self.width + self.goal_box_width) / 2),  # 4
          (0, (self.width + self.penalty_box_width) / 2),  # 5
          (0, self.width),  # 6
          (self.goal_box_length, (self.width - self.goal_box_width) / 2),  # 7
          (self.goal_box_length, (self.width + self.goal_box_width) / 2),  # 8
          (self.penalty_spot_distance, self.width / 2),  # 9
          (self.penalty_box_length, (self.width - self.penalty_box_width) / 2),  # 10
          (self.penalty_box_length, (self.width - self.goal_box_width) / 2),  # 11
          (self.penalty_box_length, (self.width + self.goal_box_width) / 2),  # 12
          (self.penalty_box_length, (self.width + self.penalty_box_width) / 2),  # 13
          (self.length / 2, 0),  # 14
          (self.length / 2, self.width / 2 - self.centre_circle_radius),  # 15
          (self.length / 2, self.width / 2 + self.centre_circle_radius),  # 16
          (self.length / 2, self.width),  # 17
          (
              self.length - self.penalty_box_length,
              (self.width - self.penalty_box_width) / 2
          ),  # 18
          (
              self.length - self.penalty_box_length,
              (self.width - self.goal_box_width) / 2
          ),  # 19
          (
              self.length - self.penalty_box_length,
              (self.width + self.goal_box_width) / 2
          ),  # 20
          (
              self.length - self.penalty_box_length,
              (self.width + self.penalty_box_width) / 2
          ),  # 21
          (self.length - self.penalty_spot_distance, self.width / 2),  # 22
          (
              self.length - self.goal_box_length,
              (self.width - self.goal_box_width) / 2
          ),  # 23
          (
              self.length - self.goal_box_length,
              (self.width + self.goal_box_width) / 2
          ),  # 24
          (self.length, 0),  # 25
          (self.length, (self.width - self.penalty_box_width) / 2),  # 26
          (self.length, (self.width - self.goal_box_width) / 2),  # 27
          (self.length, (self.width + self.goal_box_width) / 2),  # 28
          (self.length, (self.width + self.penalty_box_width) / 2),  # 29
          (self.length, self.width),  # 30
          (self.length / 2 - self.centre_circle_radius, self.width / 2),  # 31
          (self.length / 2 + self.centre_circle_radius, self.width / 2),  # 32
      ]