% Generate cone mosaic
% SL Dec 2025

ieInit;

%% scanning parameters
rsp_x_deg = 15;
rsp_y_deg = 15;
ecc_x_deg = 0;
ecc_y_deg = 0;
%% sample mosaic
% This method crops a region from a large (45x45 deg) precomputed
% lattice. This is the fastest way to generate a @cMosaic at any
% eccentricity
cm = cMosaic(...
    'size degs', [rsp_x_deg rsp_y_deg], ...            % SIZE: x=4.0 degs, y=3.0 degs
    'position degs', [ecc_x_deg ecc_y_deg] ...  % ECC:  x=20 deg, y= -15 deg, near the edge of the precomputed 45x45 mosaic
    );
%% calculate outer sgement length
eccDegs = sqrt(sum(cm.coneRFpositionsDegs .^2, 2));
[osl_microns, osl_foveal_microns] = cMosaic.outerSegmentLengthFromEccentricity(eccDegs);
jittered_seed = randi(99999);
rng(jittered_seed)
sigma = 0.2 * ones(size(osl_microns));
osl_jittered = osl_microns + sigma .* randn(size(osl_microns));
%% extract variables of interest
mosaic = {};  % empty structure for saving

% extract variables of interest
% ERICA uses Y,X while ISETBIO uses X,Y. 
mosaic.rsp_deg_y_x = [rsp_y_deg, rsp_x_deg];
mosaic.ecc_deg_y_x = [ecc_y_deg, ecc_x_deg];
mosaic.microns_per_deg = cm.micronsPerDegree;
mosaic.cone_position_um = cm.coneRFpositionsMicrons(:,[2,1]);
mosaic.cone_position_deg = cm.coneRFpositionsDegs(:,[2,1]);
mosaic.cone_spacings_deg = cm.coneRFspacingsDegs;
mosaic.cone_spacings_um = cm.coneRFspacingsMicrons;
mosaic.cone_types = cm.coneTypes;
mosaic.cone_ratios = cm.coneDensities;
mosaic.cone_aperture_um = cm.coneApertureDiametersMicrons;
mosaic.aperture2gaussian_sigma = cm.coneApertureToConeCharacteristicRadiusConversionFactor;
mosaic.aperture2diameter = cm.coneApertureToDiameterRatio;
mosaic.osl_um = osl_jittered;
mosaic.osl_jitter_seed = jittered_seed;
%% check OSL and two beam interference
figure('Position',[0,20 500, 500])
scatter(mosaic.cone_position_deg(:,2), mosaic.cone_position_deg(:,1), 1, osl_jittered);
colormap('turbo')
colorbar
xlabel('Horizontal Position (deg)')
ylabel('Vertical Position (deg)')
title('Outer Segment Length (um)')
 
n = 1.430;
lambda = 830;
opd_nm = 2*n*osl_jittered*1e3;
d_phi = 2*pi*opd_nm / lambda;
reflectance = 0.5 * (1+cos(d_phi));
figure('Position',[1000,20 500, 500])
scatter(mosaic.cone_position_deg(:,2), mosaic.cone_position_deg(:,1), 1, reflectance)
colormap("gray")
colorbar
xlabel('Horizontal Position (deg)')
ylabel('Vertical Position (deg)')
title('Reflectance From Two Beam Interference')
%% saving
savefolder = "D:\Sirui\project\AOSLO_ORG\Code\simulation\isetbio_retina";

% save mosaic
prefix = "mosaic_";
suffix = "mat";
[fullpath, fname] = autoname(savefolder, prefix, suffix);
save(fullpath, "mosaic");
fprintf('Mosaic has been saved to %s\n', fullpath)

prefix = 'mosaic_full_';
[fullpath, fname] = autoname(savefolder, prefix, suffix);
save(fullpath, "cm");
fprintf('Full mosaic has been saved to %s\n', fullpath)
