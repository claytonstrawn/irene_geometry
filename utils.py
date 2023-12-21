def check_segments_intersect(ux1,ux2,uy1,uy2,vx1,vx2,vy1,vy2):
    m_u = (uy2-uy1)/(ux2-ux1)
    b_u = uy2-m_u*ux2
    m_v = (vy2-vy1)/(vx2-vx1)
    b_v = vy2-m_v*vx2
    x_int = -(b_v-b_u)/(m_v-m_u)
    if ux1<=x_int and ux2>=x_int and vx1<=x_int and vx2>=x_int:
        return x_int
    else:
        return None