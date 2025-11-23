# diagnose_kernel.py
import sys, json, numpy as np
from src.kernels import U_ananta_hybrid
from src.fft_pipeline import conv_fft, gradient_from_phi
from src.newtonian import phi_newtonian_from_rho, G
from run_sparc_lite import safe_two_component_disk, choose_box_and_grid, get_U_grid

gal_name = sys.argv[1] if len(sys.argv)>1 else "NGC3198"

# Load gal props from your data CSV or from the NIGHTMARE_FLEET in run_sparc_lite
import csv
def read_gal_props(name, csvpath="data/galaxies.csv"):
    try:
        with open(csvpath, newline='', encoding='utf-8') as f:
            for r in csv.DictReader(f):
                if r['name']==name:
                    return {
                        'name':name,
                        'Rd_star':float(r['Rd_star_kpc']),
                        'Mstar':float(r['Mstar_Msun']),
                        'hz_star':float(r.get('hz_star_kpc',0.3)),
                        'Rd_gas':float(r.get('Rd_gas_kpc',1.8*float(r['Rd_star_kpc']))),
                        'Mgas':float(r.get('Mgas_Msun',0.0)),
                        'hz_gas':float(r.get('hz_gas_kpc',0.15))
                    }
    except Exception as e:
        pass
    raise SystemExit("Galaxy not found in CSV. Use NIGHTMARE_FLEET fallback or pass full props.")

gal = read_gal_props(gal_name)

# choose L, mu you recently used (or read from results file)
with open("results/all_galaxy_params.json",'r') as f:
    params = json.load(f).get(gal_name)
L = params['L']; mu = params['mu']

print(f"Diagnosing {gal_name} with L={L}, mu={mu}")

# grid
R_obs_approx = 30.0
Lbox, n = choose_box_and_grid(R_obs_approx, L)
print("Lbox, n, dx:", Lbox, n, (2*Lbox)/n)

# build rho
rho, dx = safe_two_component_disk(n, Lbox, gal['Rd_star'], gal['Mstar'], gal['hz_star'],
                                  gal['Rd_gas'], gal['Mgas'], gal['hz_gas'])

# Newtonian phi and grads
phi_b = phi_newtonian_from_rho(rho, Lbox, Gval=np.float32(G))
gx_b, gy_b, gz_b = gradient_from_phi(phi_b, Lbox)

# Kernel
U = get_U_grid(n, Lbox, L, "ananta-hybrid")
print("U stats: min/max/mean:", U.min(), U.max(), float(U.mean()))

phi_K_raw = conv_fft(rho, U, zero_mode=True)
print("phi_K_raw stats:", float(phi_K_raw.min()), float(phi_K_raw.max()), float(np.median(phi_K_raw)))

phi_K = mu * np.float32(G) * phi_K_raw
print("phi_K (mu*G*phi_K_raw) stats:", float(phi_K.min()), float(phi_K.max()))

gx_K, gy_K, gz_K = gradient_from_phi(phi_K, Lbox)
print("gx_K stats:", float(gx_K.min()), float(gx_K.max()), "mean:", float(gx_K.mean()))

# Compare signs with baryons at midplane (z index)
iz = n//2
g_baryon = np.sqrt(gx_b[:,:,iz]**2 + gy_b[:,:,iz]**2)
g_kernel = np.sqrt(gx_K[:,:,iz]**2 + gy_K[:,:,iz]**2)
print("g_baryon stats center-slice:", float(g_baryon.min()), float(g_baryon.max()), float(g_baryon.mean()))
print("g_kernel stats center-slice:", float(g_kernel.min()), float(g_kernel.max()), float(g_kernel.mean()))

# Check sign correlation in radial bins (are gx_K and gx_b aligned or opposite?)
mask = np.abs(gx_b[:,:,iz])>0
sign_agreement = np.mean(np.sign(gx_b[:,:,iz][mask]) == np.sign(gx_K[:,:,iz][mask]))
print("sign-agreement fraction (gx):", sign_agreement)

# Estimate v_kernel from g_kernel radial profile
R = np.linspace(0.5*dx, Lbox, 30)
from run_sparc_lite import radial_profile_2d
centers, g_mean_kernel = radial_profile_2d(np.sqrt(gx_K[:,:,iz]**2 + gy_K[:,:,iz]**2), dx, R.max(), nbins=30)
v_kernel_est = np.sqrt(np.maximum(centers * g_mean_kernel, 0.0))
print("v_kernel_est (first 6):", v_kernel_est[:6].tolist(), " max:", float(np.max(v_kernel_est)))
