import numpy as np
size = 128

def pixtocoord(i, j): # pixel (0,0) starting from left top -> coord (0,0) at the center
    return i-size//2+0.5, j+size//2-0.5

def coordtopix(x, y): # coord (0,0) at the center -> pixel (0,0) starting from left top
    return int(x)-size//2, int(y)+size//2

def rotate(original_image, depth_map, Ax, Ay, dx, dy, dz, t, a):
    '''
    input parameters:
        original image : RGB image input
        depth_map : D image input - after passing the depth estimation layer
        Ax, Ay : Field(Angle) of View - two variables will be estimated by the layer
        dx, dy, dz : camera moving at x, y, z-axes; z is the way the camera sees,
                     and (x, y) is the image pixel (unit of depth = unit of pixel here)
        t, a : theta, alpha for rotating camera

    output parameters:
        tr2_image: RGBD map on 2D image
    '''
    cost = np.cos(t); sint = np.sin(t)
    cosa = np.cos(a); sina = np.sin(a)
    tanAx = np.tan(Ax); tanAy = np.tan(Ay)
    
    # transform to 3D
    tr3_image = []
    for i in range(len(size)):
        for j in range(len(size)):
            x, y = pixtocoord(i, j)
            z = depth_map[i][j] / np.sqrt(x**2 * tanAx + y**2 * tanAy + 1)
            x -= dx; y -= dy; z -= dz
            
            tr3_image.append([((x * cost - z * sint) * (cosa + y / np.sqrt(x**2 + z**2) * sina),
                              y * cosa - np.sqrt(x**2 + z**2) * sina,
                              (z * cost + x * sint) * (y / np.sqrt(x**2 + z**2) * cosa - sina)),
                              list(original_image[i][j])]) #[(x,y,z)

    # transform to 2D
    tr2_rgb = np.zeros([size, size, 3])
    tr2_depth = np.empty([size, size, 1]).fill(-1)
    tr2_image = np.concatenate((tr2_rgb, tr2_depth), axis=2)
    for pix in range(len(tr3_image)):
        x, y, z = tr3_image[pix][0][0], tr3_image[pix][0][1], tr3_image[pix][0][2]
        x, y, r = int(x / (z * tanAx) + 0.5), int(y / (z * tanAy) + 0.5), np.sqrt(x**2 + y**2 + z**2)
        i, j = coordtopix(x, y)
        if 0 <= i < size and 0 <= j < size:
            if tr2_image[i][j][3] == -1 or tr2_image[i][j][3] > r:
                tr2_image[i][j] = tr3_image[pix][1] + [r]
    
    return tr2_image
