import magbox
import numpy as np
import os
import pytest

testdata = [ # N, K, J, dt ,T
    (256,1.0,0.5,1,400),
    (512,1.0,0.5,1,400),
    (1024,1.0,0.5,1,400),
    (256,1.0,1.0,0.5,400),
    (256,1.0,2.0,0.1,400),
    (256,1.0,3.0,0.1,400),
]

def plot_fun(t,x,y,ft_abs,q,w,K,J,dispersion,dispersion_theory,err,mean_err,max_err):
    if "PYTEST_CURRENT_TEST" in os.environ:
        print("测试模式不作图")
        return
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,10))

    plt.subplot(2,2,1)
    plt.imshow(ft_abs.T,
            origin='lower',
            extent=(q[0],q[-1],w[0],w[-1]),
    )
    plt.ylim(0,1.5*(K+2*J))

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
def test_spin_chain(N,K,J,dt,T):
    # N=256
    # K=1.0
    # J=0.5
    # dt=1
    # T=50
    rng=np.random.default_rng()
    theta0=np.ones(N)*0.01
    phi0=rng.random(N)*2*np.pi
    # phi0=10*np.arange(N)/N *2*np.pi
    LT = magbox.Lattice(type="square", size=[N], periodic=True)
    vars = magbox.Vars(K1=K, J=J)
    dispersion_fun=lambda qf: K+J*(1-np.cos(qf))*np.cos(np.mean(theta0))

    spin=magbox.spin(theta0,phi0,LT,device='cpu',type='f64')
    sf=magbox.llg(spin,vars,alpha=0,T=T,dt=dt)

    t_tc,ang,stats,err_info=sf.run(spin)
    t=t_tc.cpu().detach().numpy()
    theta=ang[::2].cpu().detach().numpy()
    phi=ang[1::2].cpu().detach().numpy()

    x=np.sin(theta)*np.cos(phi)
    y=np.sin(theta)*np.sin(phi)
    z=np.cos(theta)

    u=x+1j*y
    ft=np.fft.fft2(u)
    ft_abs=np.abs(ft)
    w=np.fft.fftfreq(len(t), dt)*2*np.pi
    q=np.fft.fftfreq(N,1)*2*np.pi

    ft_abs=np.fft.fftshift(ft_abs)
    w=np.fft.fftshift(w)
    q=np.fft.fftshift(q)

    dispersion=np.zeros(len(q))
    for idx in range(len(q)):
        arg_max=np.argmax(ft_abs[idx,:])
        dispersion[idx]=w[arg_max]
    dispersion_theory=dispersion_fun(q)
    err=dispersion/dispersion_theory-1
    max_err=np.max(np.abs(err))
    mean_err=np.mean(np.abs(err))

    print(f"mean error: {mean_err:.2e}, max err: {max_err:.2e}")

    plot_fun(t,x,y,ft_abs,q,w,K,J,dispersion,dispersion_theory,err,mean_err,max_err)

    assert mean_err<1e-2

if __name__=="__main__":
    test_spin_chain(256,3.0,3.5,0.1,300)


# spin_chain_test(256,1,0.5,1,50)
