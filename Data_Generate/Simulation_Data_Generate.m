function [B,s_real,Result] = Simulation_Data_Generate (LFM,Cortex,Time,OPTIONS)

Def_OPTIONS.WGN         = 1;
Def_OPTIONS.uniform     = 1;
Def_OPTIONS.ar          = 0;
Def_OPTIONS.GridLoc     = [];
Def_OPTIONS.MixMatrix = eye(4);
OPTIONS = struct_copy_fields(OPTIONS, Def_OPTIONS, 0);

GridLoc = OPTIONS.GridLoc;
AreaDef = OPTIONS.DefinedArea;
seedvox = OPTIONS.seedvox;
f = OPTIONS.frequency;
tau = OPTIONS.tau;
omega = OPTIONS.omega;
A = OPTIONS.Amp;
Uniform = OPTIONS.uniform;
WGN = OPTIONS.WGN;
SNR = OPTIONS.SNR;
SNIR = OPTIONS.SNIR;
ar = OPTIONS.ar;
MixMatrix = OPTIONS.MixMatrix;
nSource = size(LFM,2);
ActiveVoxSeed = num2cell(seedvox);
ActiveVox = [];
[~, VertArea] = tess_area(Cortex.Vertices, Cortex.Faces);
Cortex.VertConn = tess_vertconn(Cortex.Vertices, Cortex.Faces);
for k = 1:numel(seedvox)
    ActiveVoxSeed{k} = PatchGenerate(seedvox(k),Cortex.VertConn,VertArea,AreaDef(k));
    ActiveVox = union(ActiveVoxSeed{k},ActiveVox);
end
Area = sum(VertArea(ActiveVox));
StimTime = find(abs(Time) == min(abs(Time)));
s_real = zeros(nSource,numel(Time));
Activetime = StimTime+1:numel(Time);
if ~ar
    Basis = zeros(numel(tau),numel(Time));
    for k = 1:numel(tau)
        Basis(k,Activetime) = sin(2*pi*f(k)*(Time(Activetime))).*exp(-((Time(Activetime)-tau(k))/omega(k)).^2);
    end
    Basis = orth(Basis')';
    AA = MixMatrix*A;
    if Uniform
        for k = 1:numel(seedvox)
            s_real(ActiveVoxSeed{k},:) = repmat(AA(k,:)*Basis,numel(ActiveVoxSeed{k}),1);
        end
    else
        for k = 1:numel(seedvox)
            dis = sqrt(sum((GridLoc(ActiveVoxSeed{k},:)-repmat(GridLoc(seedvox(k),:),numel(ActiveVoxSeed{k}),1)).^2,2));
            Amplitude = exp(-dis.^2./(0.6*max(dis)).^2);
            s_real(ActiveVoxSeed{k},:) = Amplitude*AA(k,:)*Basis;
        end
    end

else
    cfg             = [];
    cfg.ntrials     = 1;
    cfg.triallength = max(Time);
    cfg.fsample     = 1/(Time(2) - Time(1));
    cfg.nsignal     = size(OPTIONS.noisecov,1);
    cfg.method      = 'ar';

    cfg.params      = OPTIONS.params;
    cfg.noisecov    = OPTIONS.noisecov;

    artimeseries              = ft_connectivitysimulation(cfg);
    s_active = zeros(numel(seedvox),numel(Time));
    gen_data = artimeseries.trial{1}(1:numel(seedvox),:);

    target_len = length(Activetime);

    s_active(:,Activetime) = gen_data(:, 1:target_len);
    for k = 1:numel(seedvox)
        s_real(ActiveVoxSeed{k},:) = A*repmat(s_active(k,:),numel(ActiveVoxSeed{k}),1);
    end
end
s_real = s_real ./ norm(s_real, 'fro');

flag = 1;
while flag
    ind = randperm(nSource);
    pink_nclus =5;
    pink_Area = unifrnd(3,5,1,pink_nclus)*1e-4;
    pink_seedvox = ind(1:pink_nclus);
    pink_ActiveVoxSeed = num2cell(pink_seedvox);
    pink_ActiveVox = [];
     for k = 1:numel(pink_seedvox)
            pink_ActiveVoxSeed{k} = PatchGenerate(pink_seedvox(k),Cortex.VertConn,VertArea,pink_Area(k));
            pink_ActiveVox = union(pink_ActiveVoxSeed{k},pink_ActiveVox);
     end
     flag = numel(intersect(pink_ActiveVox,ActiveVox));
end
pn = mkpinknoise(numel(Time),numel(pink_ActiveVox))';
EEG_brain_noise = zeros(nSource,numel(Time));
EEG_brain_noise(pink_ActiveVox,:) = pn;
EEG_brain_noise = EEG_brain_noise ./ norm(EEG_brain_noise,'fro');
ri = 1/10^(SNIR/10);
EEG_brain_signal = s_real + ri*EEG_brain_noise;
EEG_sensor_signal = LFM*EEG_brain_signal;

if  ~WGN
    load 'ArtHumanEEGNoise.mat'
    noise = noise(:,1:size(s_real,2));
    r = norm(LFM*s_real,'fro')/norm(noise,'fro')/10^(SNR/20);
    Noise = r*noise;
    B = EEG_sensor_signal + Noise;
else
    B = awgn(EEG_sensor_signal,SNR,'measured');
end
Result.B = B;
Result.real = s_real;
Result.seedvox = seedvox;
Result.ActiveVox = ActiveVox;
Result.ActiveVoxSeed = ActiveVoxSeed;
Result.Area = Area;

function noise= mkpinknoise(n,m)

n1=2*ceil(n/2)+1;
scal=sqrt(1./[1:(n1-3)/2]');
ff=zeros(n1,m);
ff(2:(n1-1)/2,:)=repmat(scal,1,m);
noise=fft(randn(n1,m));
noise=2*real(ifft(noise.*ff));
noise=noise(1:n,:);
