#-*- encoding: utf-8 -*-
'''
Created on 2016-5-20

@author: zhangdan
'''
from numpy import *

def function_to_minimise(p):
    r = array([10*(p[1]-p[0]**2),(1-p[0])])
    fp = dot(transpose(r),r) #= 100*(p[1]-p[0]**2)**2 + (1-p[0])**2
    J = (array([[-20*p[0],10],[-1,0]]))
    grad = dot(transpose(J),transpose(r))
    return fp,r,grad,J

def lm(p0,tol=10**(-5),maxits=100):
    
    nvars=shape(p0)[0]
    nu=0.01
    p = p0
    fp,r,grad,J = function_to_minimise(p)
    e = sum(dot(transpose(r),r))
    nits = 0
    while nits<maxits and linalg.norm(grad)>tol:
        nits += 1
        fp,r,grad,J = function_to_minimise(p)
        H=dot(transpose(J),J) + nu*eye(nvars)

        pnew = zeros(shape(p))
        nits2 = 0
        while (p!=pnew).all() and nits2<maxits:
            nits2 += 1
            dp,resid,rank,s = linalg.lstsq(H,grad)
            pnew = p - dp
            fpnew,rnew,gradnew,Jnew = function_to_minimise(pnew)
            enew = sum(dot(transpose(rnew),rnew))
            rho = linalg.norm(dot(transpose(r),r)-dot(transpose(rnew),rnew))
            rho /= linalg.norm(dot(transpose(grad),pnew-p))
            
            if rho>0:
                update = 1
                p = pnew
                e = enew
                if rho>0.25:
                    nu=nu/10
            else: 
                nu=nu*10
                update = 0
        
        print fp, p, e, linalg.norm(grad), nu
       #return fp, p, linalg.norm(grad), nu

if __name__ == '__main__': 
    set_printoptions(formatter={'float': '{: 0.3f}'.format})   
    p0 = random.randn(2)
    #f_x, para, grad_norm, nu = lm(p0)
    lm(p0)