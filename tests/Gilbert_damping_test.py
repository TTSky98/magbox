import magbox
import numpy as np
import os
import pytest
import scipy

testdata = [ # N, K, J, alpha
    (256, 2.0, 2.0, 0.1),
    (256, 3.0, 8.0, 0.1),
    (256, 1.0, 2.0, 0.01),
    (256, 1.0, 2.0, 0.05),
    (128, 1.0, 2.0, 0.05),
]

def plot_fun(N,err,mean_err,max_err):
    if "PYTEST_CURRENT_TEST" in os.environ:
        print("测试模式不作图")
        return
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,10))

    # plt.subplot(2,2,1)
    # plt.imshow(en,
    #         origin='lower',
    #         extent=(t[0],t[-1],1,N),
    # )
    # plt.xlabel('t')
    # plt.ylabel('index')
    
    # plt.subplot(2,2,2)
    # err_tmp=err.reshape(-1)
    plt.scatter(np.linspace(1,N,N),err,label="Error")
    plt.legend()
    plt.title(f"mean error: {mean_err:.2e}, max err: {max_err:.2e}")

    # plt.subplot(2,2,3)
    # plt.legend()
    # plt.xlabel('t')
    # plt.ylabel('energy')
    # plt.title(f"index: {id},mean err: {err_t[id]:.2e}")
    plt.show()

def peak_correction(left_height,right_height,peak_height):
    if left_height > right_height:
        h1=left_height
        loc=-1
    else:
        h1=right_height
        loc=1
    epsilon=h1/(h1+peak_height)
    
    return loc*epsilon

# @pytest.skip(reason="Under rebuild")
@pytest.mark.parametrize("N,K,J,alpha",testdata)
def test_energy_damping(N,K,J,alpha):
    # N=256
    # K=1.0
    # J=0.5
    # dt=1
    # T=50
    # rng=np.random.default_rng()
    dt=2*np.pi/(K+2*J)
    loops=np.floor(1/alpha)*100
    T=np.min([np.ceil(loops*dt),2000.0])

    dt=T/np.ceil(T/dt)
    dt=dt/8
    theta0=np.ones(N)*0.01
    phi0=np.random.rand(N)*2*np.pi
    
    LT = magbox.Lattice(type="square", size=[N], periodic=True)
    vars = magbox.Vars(K1=K, J=J)

    spin=magbox.spin(theta0,phi0,LT,device='cpu',type='f64')
    sf=magbox.llg(spin,vars,alpha=alpha,T=T,dt=dt)

    t_tc,ang,*_=sf.run(spin)
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
    dw=w[1]-w[0]
    q=np.fft.fftshift(q)

    alpha_q=np.zeros(len(q))
    alpha_qc=np.zeros(len(q))
    for idx in range(len(q)):
        peaks,property=scipy.signal.find_peaks(ft_abs[idx,:]**2,width=(None,None))
        peak_idx=np.argmax(property['prominences'])
        omega=w[peaks[peak_idx]]
        omega_correction=peak_correction(ft_abs[idx,peaks[peak_idx]-1],ft_abs[idx,peaks[peak_idx]+1],ft_abs[idx,peaks[peak_idx]])*dw
        width=property['widths'][peak_idx]*dw
        alpha_q[idx]=width/omega/2
        alpha_qc[idx]=width/2/(omega-omega_correction)
    err=alpha_q/alpha-1
    mean_err=np.mean(np.abs(err))
    max_err=np.max(np.abs(err))

    err_c=alpha_qc/alpha-1
    mean_errc=np.mean(np.abs(err_c))
    max_errc=np.max(np.abs(err_c))
    

    print(f"mean error: {mean_err:.2e}, max err: {max_err:.2e}")
    print(f"mean error correction: {mean_errc:.2e}, max err correction: {max_errc:.2e}")
    print(f"Technical accuracy: {2*dw/K:.2e}")

    plot_fun(N,err,mean_err,max_err)

    assert mean_err<2*dw/K

if __name__=="__main__":
    test_energy_damping(32, 1.0, 5.0, 0.1)


# spin_chain_test(256,1,0.5,1,50)
