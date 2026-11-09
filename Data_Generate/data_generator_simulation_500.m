clc
clear
tic

algorithms = {'VSSI-LpR', 'VSSI-L1', 'wMNE', 'LORETA', 'LPSSO', 'LASSO'};
ResultsLoading = [0 0 0 0 0 0];
WGN = 1;
LFNormlization = 0;
Uniform = 1;
Test = 0;
VariousExtents = 1;
VariousSNRs = 0; cv_p = 0;
VariousPatches = 0;
VariousPvalues = 0;
VariousCorrelation = 0;
VariousChannels = 0;
VariousSNIRs = 0;

variousconditions = 0;

if variousconditions+VariousExtents+VariousSNRs+VariousPatches+VariousCorrelation+VariousPvalues+VariousChannels+Test ~= 1
    error('There will be one and only one scenario.');
end
if WGN
    channelselect=[1:32,34:42,44:64];
else
    channelselect=[1:32 34:42 44:59 61:63];
end
[sStudy, iStudy] = bst_get('StudyWithCondition','Subject04/Simulation4');
index = 1;
bst_call(@export_matlab, {char(sStudy.Data(1,index).FileName)},'data');

dt = 0.002;
data.Time = ((1:500) - 151) * dt;
disp('✅ 时间轴已重写：起点 -0.3s，第 151 个点为 0s (刺激发生)');

[sSurface, iSurface] = bst_get('SurfaceFileByType',[],'Cortex');
bst_call(@export_matlab, {char(sSurface.FileName)},'Cortex');
Atlas = Cortex.Atlas(2);

[sHeadModel] = bst_get('HeadModelForStudy', iStudy);
bst_call(@export_matlab, {char(sHeadModel.FileName)},'model');
Gain=model.Gain(channelselect,:);
GridLoc=model.GridLoc;
GridOrient=model.GridOrient;

Gain = bst_gain_orient(Gain,GridOrient);
clear GridOrient
[nSensor,nSource] = size(Gain);
L = Gain;

disp('Generating neighborhood index table for PyTorch...');
adj_matrix = Cortex.VertConn;
adj_matrix = adj_matrix | adj_matrix';

num_vertices = size(Cortex.Vertices, 1);
max_neighbors = 10;
neighbors_idx = zeros(num_vertices, max_neighbors);

for v = 1:num_vertices
    neighbors = find(adj_matrix(v, :));
    num_n = length(neighbors);

    if num_n >= max_neighbors
        neighbors_idx(v, :) = neighbors(1:max_neighbors);
    elseif num_n > 0
        padding = repmat(v, 1, max_neighbors - num_n);
        neighbors_idx(v, :) = [neighbors, padding];
    else
        neighbors_idx(v, :) = repmat(v, 1, max_neighbors);
    end
end

neighbors_idx = neighbors_idx - 1;
disp('Neighborhood index table generated successfully.');

if VariousSNRs
   scenario = 'various SNRs';
   SNR1 = [-5,0,5,10];
   SNIR1 = zeros(4,1)+5;
   condition = SNR1';
   K = ones(5,1);
   DefinedArea = 8*1e-4*ones(size(condition,1),max(K));
elseif VariousSNIRs
   scenario = 'various conditions';
   SNR1 = zeros(5,1)+5;
   SNIR1 = [-5,0,5,10];
   condition = SNIR1';
   K = ones(5,1);
   DefinedArea = 8*1e-4*ones(size(condition,1),max(K));
elseif VariousExtents
   scenario = 'various Extents';
   SNR1 = 5*ones(5,1);
   SNIR1 = 5*ones(5,1);
   condition = [2 5 10 18 32]';
   K = ones(5,1);
   DefinedArea = [2 5 10 18 32]'*1e-4*ones(1,max(K));
elseif VariousChannels
   scenario = 'various channels';
   SNR1 = 5*ones(4,1);
   SNIR1 = 5*ones(4,1);
   condition = [62, 46, 32, 16]';
   K = ones(4,1);
   DefinedArea = 8*1e-4*ones(size(condition,1),max(K));
elseif VariousPatches
   scenario = 'various patches';
   SNR1 = 5*ones(4,1);
   SNIR1 = 5*ones(4,1);
   condition = [1:4]';
   K = [1,2,3,4];
   DefinedArea = 8*1e-4*ones(size(condition,1),max(K));
elseif variousconditions
    scenario = 'various conditions';
    SNR1 = [-5,0,5,10];
    SNIR1 = [-5,0,5,10];
    condition = SNR1';
    K = [1,2,3,4,5];
    DefinedArea = [2 5 10 18 32]'*1e-4*ones(1,max(K));

elseif Test
    algorithms = {'SSSI-L2p'}
    ResultsLoading = [0 0 0];
    scenario = 'test';
    SNR1 = 0;
    SNIR1 = 0;
    condition = [1];
    K = 1;
    DefinedArea = 5*1e-4;
end

Miter = 125;
outpath = 'F:\VSSI-ARD-net\Data\ns_6003_miter_10000_simple';
if variousconditions
    set = 'train';
elseif Test
    set = 'test';
else
    set = 'test';
end
outpath = fullfile(outpath, set);

for i = 1 : size(condition,1)
    if variousconditions
        path{i} = fullfile(outpath,scenario);
    else
        path{i} = fullfile(outpath,scenario,'\',num2str(condition(i)));
    end
    if ~exist(path{i})
        mkdir(path{i});
    end
end

Eccentricity = sqrt(sum(GridLoc.^2,2));
L_norm = sqrt(sum(L.^2, 1))';
threshold_L = prctile(L_norm, 60);
y_abs_limit = 0.005;
Ec = find(Eccentricity > 50*1e-3 & abs(GridLoc(:, 2)) > y_abs_limit & L_norm > threshold_L);
if numel(Ec) < numel(GridLoc) * 0.1
    warning('筛选条件过严，正在放宽源点限制...');
    Ec = find(L_norm > prctile(L_norm, 45));
end

fixed_seedvox_pool = Ec(randi(numel(Ec), Miter, 1));

for h = 1:2

for iteration = 1:size(condition,1)
    se = 1;

    B_totalSize = [Miter, 62, 6];
    s_real_totalSize = [Miter,6003,6];
    TBFs_totalSize = [Miter,6,500];
    ActiveVoxSeed_dataStorage={Miter,1};
    seedvox_dataStorage=zeros(Miter,1);

    B_dataStorage = zeros(B_totalSize);
    s_real_dataStorage = zeros(s_real_totalSize);
    TBFs_dataStorage = zeros(TBFs_totalSize);
    for iter = 1:Miter
        if variousconditions
            ind_SNR = randi([1, numel(SNR1)]);
            ind_SNIR = randi([1, numel(SNIR1)]);
            ind_extent = randi([1, size(DefinedArea,1)]);
            SNR = SNR1(ind_SNR);
            SNIR = SNIR1(ind_SNIR);
        else
            ind_extent = iteration;
            if VariousSNRs
                SNR = condition(iteration);
                SNIR = SNIR1(1);
            elseif VariousSNIRs
                SNR = SNR1(1);
                SNIR = condition(iteration);
            elseif VariousExtents
                SNR = SNR1(1);
                SNIR = SNIR1(1);
            else
                SNR = SNR1(1);
                SNIR = SNIR1(1);
            end
        end

        seedvox = fixed_seedvox_pool(iter);
        tau = [0.1 0.35 0.5 0.6];omega = [0.1 0.15 0.15 0.15];
        f = [10 11 8 9];
        Amp = 1e-8;
        StimTime = 150;
        TimeLen = 500;
        Time = data.Time(1:TimeLen);

        OPTIONS.DefinedArea    = DefinedArea(ind_extent,:);
        OPTIONS.seedvox        = seedvox;
        OPTIONS.frequency      = f;
        OPTIONS.tau            = tau;
        OPTIONS.omega          = omega;
        OPTIONS.Amp            = Amp;
        OPTIONS.GridLoc        = GridLoc;
        if VariousPatches
        OPTIONS.MixMatrix = [1    0    0     0;
                             0    1    0     0;
                             0    0    1     0;
                             0    0    0     1;
                             0    0    .5     .5];
        elseif VariousCorrelation
            xi = condition(iteration);
        OPTIONS.MixMatrix = [1            0            0          0;
                             xi     sqrt(1-xi^2)       0          0;
                             xi           0        sqrt(1-xi^2)   0;
                             0            0            1          0];
        end
        OPTIONS.uniform       = Uniform;
        OPTIONS.WGN           = WGN;

        OPTIONS.SNR           = SNR;
        OPTIONS.SNIR          = SNIR;

        OPTIONS.ar            = 0;
        OPTIONS.params(:,:,1) = [ 0.8    0    0 ;
                                    0  0.9  0.5 ;
                                  0.4    0  0.5];

        OPTIONS.params(:,:,2) = [-0.5    0    0 ;
                                    0 -0.8    0 ;
                                    0    0 -0.2];

        OPTIONS.noisecov      = [ 0.3    0    0 ;
                                    0    1    0 ;
                                    0    0  0.2];
        OPTIONS.SinglePoint   = 0;

        [B,s_real,Result] = Simulation_Data_Generate(L,Cortex,Time,OPTIONS);

        ratio = 1;iteration

        ActiveVoxSeed = Result.ActiveVoxSeed;
        seedvox = Result.seedvox;

        fprintf('Actual SNR is %g\n',20*log10(norm(L*s_real,'fro')/norm(B-L*s_real,'fro')));

        B_amp = B./ratio;
        s_real_amp = s_real./ratio;
        [junk,v,d] = svd(B,'econ');
        Eigvals = diag(v).^2;
        NEigvals = Eigvals./sum(Eigvals);
        KSVD = 6;
        TBFs = d(:,1:KSVD)';

        B_trans = B_amp*TBFs';
        s_real_trans = s_real_amp*TBFs';

        V = VariationEdge(Cortex.VertConn);
        save([path{iteration} '\V.mat'], "V")

        B_dataStorage(se,:,:) = B_trans;
        s_real_dataStorage(se,:,:) = s_real_trans;
        TBFs_dataStorage(se,:,:) = TBFs;
        ActiveVoxSeed_dataStorage{se}=ActiveVoxSeed{1};
        seedvox_dataStorage(se) = seedvox;

        folderpath = path{iteration};
        filenames = ["GridLoc.mat";"Cortex.mat";"L.mat";"model.mat";"neighbors_idx.mat"];
        filenames2 = ["GridLoc";"Cortex";"L";"model";"neighbors_idx"];
        for a = 1:length(filenames)
            fullpath = fullfile(path(iteration),filenames(a))
            if exist('fullpath', 'file') == 2
                continue;
            else
                save(char(fullpath),char(filenames2(a)))
            end
        end
        se = se + 1;
    end
save([path{iteration} '\' 'datayu_' num2str(4*(h-1)+iteration) '.mat'],'ActiveVoxSeed_dataStorage','B_dataStorage','s_real_dataStorage','TBFs_dataStorage','seedvox_dataStorage');
end
end
