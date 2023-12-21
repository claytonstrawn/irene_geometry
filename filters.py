import numpy as np

def normal_curve(xs,ys,r):
    yprimes = np.gradient(ys,xs)
    norm = np.sqrt(yprimes**2+1)
    gamma_xs = xs - r*yprimes/norm
    gamma_ys = ys + r/norm
    return sanitize_gammas(xs,ys,gamma_xs,gamma_ys)

def poid_finder_selfintersect(xs,ys,r):
    nxs,nys = normal_curve(xs,ys,r)
    gxs,gys = find_top_branch(nxs,nys)
    return xs,reinterpolate_onto_orig_points(xs,gxs,gys)
    
def sanitize_gammas(xs,ys,gamma_xs,gamma_ys):
    i=0
    while gamma_xs[i]<xs[0] or gamma_xs[i]<gamma_xs[i-1]:
        i+=1
    j=len(gamma_xs)-1
    while gamma_xs[j]>xs[-1]:
        j-=1
    return gamma_xs[i:j],gamma_ys[i:j]

def find_top_branch(gamma_xs,gamma_ys):
    assert len(gamma_xs) == len(gamma_ys)
    diff = np.diff(gamma_xs)
    increasing = True
    last_x = gamma_xs[0]
    output_xs = [gamma_xs[0]]
    output_ys = [gamma_ys[0]]
    for i in range(1,len(gamma_xs)):
        if gamma_xs[i] > last_x and increasing:
            last_x = gamma_xs[i]
            if gamma_xs[i]>output_xs[-1]:
                output_xs.append(gamma_xs[i])
                output_ys.append(gamma_ys[i])
        elif gamma_xs[i] <= last_x and increasing:
            last_x = gamma_xs[i]
            increasing = False
            stop_point = i
        elif gamma_xs[i] <= last_x and not increasing:
            last_x = gamma_xs[i]
        elif gamma_xs[i] > last_x and not increasing:
            last_x = gamma_xs[i]
            increasing = True
            start_point = i
            n_to_delete,uv_x,uv_y = find_intersection(gamma_xs,gamma_ys,start_point,stop_point)
            current_output_len = len(output_xs)
            output_xs = output_xs[:current_output_len-n_to_delete]
            output_ys = output_ys[:current_output_len-n_to_delete]
            output_xs += list(uv_x)
            output_ys += list(uv_y)
    return np.array(output_xs),np.array(output_ys)
            
def find_intersection(gamma_xs,gamma_ys,start_point,stop_point):
    left_edge = gamma_xs[start_point]
    right_edge = gamma_xs[stop_point]
    left_points_xs = gamma_xs[:stop_point]
    left_points_ys = gamma_ys[:stop_point]
    u_xs = left_points_xs[left_points_xs>=left_edge]
    u_ys = left_points_ys[left_points_xs>=left_edge]
    right_points_xs = gamma_xs[start_point:]
    right_points_ys = gamma_ys[start_point:]
    v_xs = right_points_xs[right_points_xs<=right_edge]
    v_ys = right_points_ys[right_points_xs<=right_edge]
    for i in range(len(u_xs)-1):
        ux1,uy1 = u_xs[i],u_ys[i]
        ux2,uy2 = u_xs[i+1],u_ys[i+1]
        for j in range(len(v_xs)-1):
            vx1,vy1 = v_xs[j],v_ys[j]
            vx2,vy2 = v_xs[j+1],v_ys[j+1]
            x_int = check_segments_intersect(ux1,ux2,uy1,uy2,vx1,vx2,vy1,vy2)
            if x_int is not None:
                xs_to_ret = np.concatenate((u_xs[u_xs<=x_int],v_xs[v_xs>x_int]))
                ys_to_ret = np.concatenate((u_ys[u_xs<=x_int],v_ys[v_xs>x_int]))
                return len(u_xs),xs_to_ret,ys_to_ret
    xs_to_ret = np.array([u_xs[0],v_xs[-1]])
    ys_to_ret = np.array([u_ys[0],v_ys[-1]])
    return len(u_xs),xs_to_ret,ys_to_ret                
    
def reinterpolate_onto_orig_points(xs,poidxs,poidys):
    return np.interp(xs,poidxs,poidys)

def poid_finder_ngon(xs,ys,r,n):
    angle = 2*np.pi/n
    all_ys = np.zeros((len(xs),n//4+n//4+1))
    for i in range(-n//4,n//4+1):
        theta = i*angle
        newxs,newys = one_dim_transform(xs,ys,r,theta)
        out_ys = np.interp(xs,newxs,newys)
        all_ys[:,i] = out_ys
    return xs,np.amax(all_ys,axis=1)
        
        
def one_dim_transform(xs,ys,r,theta):
    return xs+r*np.sin(theta),ys+r*np.cos(theta)
