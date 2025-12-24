import magbox
import numpy as np
import os
import pytest

testdata = [ # N, K, J, dt ,T
    (256,1.0,0.5,1),
    (512,1.0,0.5,2),
    (1024,1.0,0.5,3),
    (256,1.0,1.0,4),
    (256,1.0,2.0,5),
    (256,1.0,3.0,5),
]

def plot_fun(t,x,y,ft_abs,q,w,dispersion,dispersion_theory,err,mean_err,max_err):
    if "PYTEST_CURRENT_TEST" in os.environ:
        print("测试模式不作图")
        return
    import matplotlib.pyplot as plt
    w_max=dispersion_theory.max()
    plt.figure(figsize=(10,10))

    plt.subplot(2,2,1)
    plt.imshow(ft_abs.T,
            origin='lower',
            extent=(q[0],q[-1],w[0],w[-1]),
    )
    plt.ylim(0,1.5*w_max)

    plt.subplot(2,2,2)
    plt.scatter(q,dispersion,label="Simulation")
    plt.plot(q,dispersion_theory,label="Theory")
    plt.legend()

    plt.subplot(2,2,3)
    plt.scatter(q,err,label="Error")
    plt.legend()
    plt.title(f"mean error: {mean_err:.2e}, max err: {max_err:.2e}")

    plt.subplot(2,2,4)
    plt.plot(t,x[0,:],label="x")
    plt.plot(t,y[0,:],label="y")
    plt.legend()
    # plt.title(f"mean error: {mean_err:.2e}, max err: {max_err:.2e}")
    plt.show()

@pytest.mark.parametrize("N,K,J,dt,T",testdata)
def test_spin_chain(N,K,J,K_hard):
    # N=256
    # K=1.0
    # J=0.5
    # dt=1
    # T=50
    rng=np.random.default_rng()
    theta0=np.ones(N)*0.01
    phi0=rng.random(N)*2*np.pi
    z=np.cos(theta0)
    x=np.sin(theta0)*np.cos(phi0)
    y=np.sin(theta0)*np.sin(phi0)
    # phi0=10*np.arange(N)/N *2*np.pi
    hard_kernel = K_hard * np.array([[0,0,0],[0,1,0],[0,0,0]])
    def hard_axis_heff(kernel, cartS):
        return -cartS @ kernel[0]
    LT = magbox.Lattice(type="square", size=[N], periodic=True)
    vars = magbox.Vars(K1=K, J=J, custom_kernel=(hard_kernel,), custom_heff=hard_axis_heff)

    dispersion_fun=lambda qf: np.sqrt((K+J*(1-np.cos(qf))*np.cos(np.mean(theta0))+K_hard)*(K+J*(1-np.cos(qf))*np.cos(np.mean(theta0))))
    q=np.fft.fftfreq(N,1)*2*np.pi
    q=np.fft.fftshift(q)
    dispersion_theory=dispersion_fun(q)
    w_max=dispersion_theory.max()
    W_diff=np.min(np.abs(np.diff(dispersion_theory)))
    print(f"freq max: {w_max:.3e}, freq diff: {W_diff:.3e}")
    dt=np.max([2*np.pi/(3*w_max),0.05])
    T=np.min([int(3*2*np.pi/W_diff),2e4])
    print(f"use dt: {dt:.3e}, Total Time: {T:.3e}")

    spin=magbox.spin3(x, y, z,LT, device='cpu',dtype='f32')
    sf=magbox.llg3(spin,vars,alpha=0,T=T,dt=dt)

    t_tc,S,stats,err_info=sf.run(spin)
    t=t_tc.cpu().detach().numpy()

    x=S[::3].detach().cpu().numpy()
    y=S[1::3].detach().cpu().numpy()
    z=S[2::3].detach().cpu().numpy()

    u=x+1j*y
    ft=np.fft.fft2(u)
    ft_abs=np.abs(ft)
    w=np.fft.fftfreq(len(t), dt)*2*np.pi
    

    ft_abs=np.fft.fftshift(ft_abs)
    w=np.fft.fftshift(w)
    

    dispersion=np.zeros(len(q))
    for idx in range(len(q)):
        arg_max=np.argmax(ft_abs[idx,:])
        dispersion[idx]=w[arg_max]
    dispersion=np.abs(dispersion)

    err=dispersion/dispersion_theory-1
    max_err=np.max(np.abs(err))
    mean_err=np.mean(np.abs(err))

    print(f"mean error: {mean_err:.2e}, max err: {max_err:.2e}")

    plot_fun(t,x,y,ft_abs,q,w,dispersion,dispersion_theory,err,mean_err,max_err)

    assert mean_err<1e-2

if __name__=="__main__":
    test_spin_chain(256,1,1,10)


# spin_chain_test(256,1,0.5,1,50)
