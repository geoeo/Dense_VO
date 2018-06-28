import numpy as np

#TODO: Test this. Last line may be in wrong dimensions
def generate_3d_plane(a,b,d,pointCount,sigma):
    c = -1 # fixed
    plane_normal = np.array([a,b,c]).astype(np.float32)
    plane_normal /= np.linalg.norm(plane_normal)

    noise = np.random.normal(0, sigma, pointCount)
    xs = np.random.uniform(-10,10,pointCount)
    ys = np.random.uniform(-10,10,pointCount)
    # divided by c = -1 implicitly
    zs = plane_normal[0]*xs + plane_normal[1]*ys + np.repeat(d,pointCount)

    plane_normal_pertrubed = map(lambda x: x*plane_normal,noise)
    return (xs + plane_normal_pertrubed[0,:],ys + plane_normal_pertrubed[1,:],zs + plane_normal_pertrubed[2,:])