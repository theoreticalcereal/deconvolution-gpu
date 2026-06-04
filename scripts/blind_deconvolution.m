%% Configuration Parameters

% =========================================================================
% INJECTED BY NEXTFLOW / PYTHON WRAPPER
% The following variables are passed directly into the workspace:
% imagePath, psfPath, psfFile, background, iter
% DO NOT hardcode them here, or it will break the pipeline!
% =========================================================================

% Map the string 'psfFile' passed by Python to the cell array 'psf' expected by the script.
if exist('psfFile', 'var')
    psf{1} = psfFile;
end

% Set defaults for variables that might not be passed by the wrapper yet.
if ~exist('Cell_name', 'var')
    Cell_name = 'Top_Cell';
end
if ~exist('Cell_index', 'var')
    Cell_index = [19];
end
if ~exist('ChannelstoProcess', 'var')
    ChannelstoProcess = [0];
end
if ~exist('timepoint', 'var')
    timepoint = [];
end

%% Load PSF
for p = 1:size(psf, 2)
    filepath = fullfile(psfPath, psf{p});
    if ~isfile(filepath)
        error('Missing expected file: %s', filepath);
    end

    PSFimage = readtiffstack(filepath);

    PSFimage = double(PSFimage);
    PSFimage = PSFimage - background;
    PSFimage = abs(PSFimage);

    PSF{p} = PSFimage;
end
clear PSFimage

%% Deconvolution setup
numfolder = numel(Cell_index);
ch_number = numel(ChannelstoProcess);

for c = 1:numfolder

    % Use the plain folder name when Cell_index is empty; otherwise append the index.
    if isempty(Cell_index)
        names2 = Cell_name;
    else
        names2 = strcat(Cell_name, num2str(Cell_index(c)));
    end

    inputDir = fullfile(imagePath, names2);

    % Discover input files in the cell folder.
    files = dir(fullfile(inputDir, '*.tiff'));
    if isempty(files)
        files = dir(fullfile(inputDir, '*.tif'));
    end
    if isempty(files)
        error('No TIFF files found in %s', inputDir);
    end

    % Sort for deterministic processing.
    [~, idx] = sort({files.name});
    files = files(idx);

    numImages = numel(files);

    % Define timepoint range.
    if isempty(timepoint)
        t_st = 0;
        t_end = round(numImages / ch_number) - 1;
    else
        t_st = min(timepoint);
        t_end = max(timepoint);
    end

    for t = t_st:t_end
        for ch = 1:ch_number

            tic

            % Expected base name for the registered consistent input file.
            baseName = sprintf('CH%02d_%06d_registered_consistent', ChannelstoProcess(ch), t);

            % Try both supported extensions.
            candidates = {
                fullfile(inputDir, [baseName '.tif'])
                fullfile(inputDir, [baseName '.tiff'])
            };

            filepath = '';
            for k = 1:numel(candidates)
                if isfile(candidates{k})
                    filepath = candidates{k};
                    break;
                end
            end

            if isempty(filepath)
                error('Missing expected file: %s', baseName);
            end

            FinalImage = readtiffstack(filepath);
            mImage = size(FinalImage, 1);
            nImage = size(FinalImage, 2);
            NumberImages = size(FinalImage, 3);

            %% Deconvolution
            % Pad volume to reduce border artifacts during deconvolution.
            E1 = padarray(single(FinalImage), [20 20 20], 'symmetric');
            maxE1 = max(E1(:));
            minE1 = min(E1(:));

            psfi = single(PSF{ch});

            % Blind deconvolution estimates both the image and the PSF.
            [Dec, psfr] = deconvblind(E1, psfi, iter);

            % Crop away padding.
            Dec = Dec(21:20+mImage, 21:20+nImage, 21:20+NumberImages);

            % Rescale to original intensity range.
            Dec = (Dec - min(Dec(:))) / (max(Dec(:)) - min(Dec(:)));
            Dec = Dec .* (maxE1 - minE1) + minE1;
            Dec = uint16(Dec);

            %% Save retrieved PSF
            mx = max(psfr(:));
            if mx == 0
                error('Estimated PSF is all zeros.');
            end
            psfr = psfr ./ mx;
            psfr2 = uint16(60000 * psfr);

            PSFfolder = fullfile(dir_Dec, strcat('PSFr_', names2));
            mkdir(PSFfolder);
            PSFname = fullfile(PSFfolder, [names2 '_psfr_ch' num2str(ch) '.tif']);
            writetiffstack(psfr2, PSFname);

            %% Double blind deconvolution
            [Dec2] = deconvlucy(E1, psfr, iter);

            Dec2 = Dec2(21:20+mImage, 21:20+nImage, 21:20+NumberImages);
            Dec2 = (Dec2 - min(Dec2(:))) / (max(Dec2(:)) - min(Dec2(:)));
            Dec2 = Dec2 .* (maxE1 - minE1) + minE1;
            Dec2 = uint16(Dec2);

            %% Save double blind deconvolved image
            finalPath2 = fullfile(dir_Dec, strcat('DB2_', names2));
            mkdir(finalPath2);

            Decname2 = fullfile(finalPath2, [baseName '.tif']);
            writetiffstack(Dec2, Decname2);

            toc, disp('Done')
        end
    end
end

disp('All Done')