import math

def angle_between(v1, v2):
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
    mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
    if mag1 == 0 or mag2 == 0:
        return 0
    cos_theta = dot/(mag1 * mag2)
    angle = math.acos(max(-1, min(1, cos_theta)))
    return math.floor(math.degrees(angle))


diff = (-1,1)
angle = angle_between((-1,0), diff)
print(angle)