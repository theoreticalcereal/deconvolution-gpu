tic;

%% Processing Setup
numFolders = 1;
if ~exist('ChannelsToProcess', 'var') || isempty(ChannelsToProcess)
    ChannelsToProcess = [0];
end
numChannels = numel(ChannelsToProcess);
if ~exist('output_dir', 'var') || isempty(output_dir)
    error('output_dir was not provided by the wrapper.');
end

for c = 1:numFolders

    % Use the folder name passed from Nextflow directly.
    cellNameWithIndex = CellName;

    inputDir = fullfile(imagePath, cellNameWithIndex);

    % Discover input files in the cell folder.
    tifFiles = dir(fullfile(inputDir, '*.tif'));
    if isempty(tifFiles)
        tifFiles = dir(fullfile(inputDir, '*.tiff'));
    end
    if isempty(tifFiles)
        error('No TIFF files found in %s', inputDir);
    end

    % Sort files for deterministic processing.
    [~, idx] = sort({tifFiles.name});
    tifFiles = tifFiles(idx);

    numImages = numel(tifFiles);

    % Define timepoints range
    if ~exist('timepoints', 'var') || isempty(timepoints)
        timepoints = [];
    end
    if isempty(timepoints)
        t_start = 0;
        t_end = round(numImages / numChannels) - 1;
    else
        t_start = min(timepoints);
        t_end = max(timepoints);
    end

    % Process each timepoint and channel
    for t = t_start:t_end
        for ch = 1:numChannels
            tic;

            baseName = sprintf('CH%02d_%06d_registered_consistent', ChannelsToProcess(ch), t);

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

            Image = readtiffstack(filepath);

            [ysize, xsize, zsize] = size(Image);
            newdz = dz * cosd(angle);

            disp("Deskewing ... ");
            clear ShearImage
            cz = floor(zsize / 2) + 1;

            max_yoffset = abs(round(flip * (1 - cz) * (newdz / dx)));
            ShearImage = zeros(ysize + 2 * max_yoffset, xsize, zsize);

            tic
            for z = 1:zsize
                yoffset = round(flip * (z - cz) * (newdz / dx));
                ShearImage(yoffset + max_yoffset + 1 : ysize + yoffset + max_yoffset, :, z) = Image(:,:,z);
            end

            for z = 1:zsize-1
                ShearImage(:,:,z) = (ShearImage(:,:,z) + ShearImage(:,:,z+1)) / 2;
            end

            ShearImage = uint16(ShearImage);
            toc

            disp("Saving the shear image");
            tic
            output_size = size(ShearImage,1) * size(ShearImage,2) * size(ShearImage,3) * 2 / (1024*1024*1024);

            outputFolder = fullfile(output_dir, strcat('shear', num2str(angle), '_mlv2_', cellNameWithIndex));
            if ~isfolder(outputFolder)
                mkdir(outputFolder);
            end
            if output_size > 4
                disp(sprintf("File is larger than 4GB (%.2f GB), saving as BigTIFF format", output_size));
            end
            writetiffstack(ShearImage, fullfile(outputFolder, [baseName '.tif']));
            toc

            disp("Rotating to top view ... ");
            clear mipzy scaled_ShearImage scaled_mipzy rot_scaled_mipzy mask cropped_mipzy zy_view cropped_rotate_zy;
            mipzy = max(ShearImage, [], 2);

            figure(1)
            imagesc(mipzy); axis equal tight

            scale_x = dz * sind(angle) / dx;

            scaled_ShearImage = imresize3(ShearImage, ...
                [size(ShearImage,1), size(ShearImage,2), round(size(ShearImage,3) * scale_x)], ...
                'Method', 'linear');

            scaled_mipzy = max(scaled_ShearImage, [], 2);

            figure(2)
            imagesc(scaled_mipzy); axis equal tight

            rot_scaled_mipzy = imrotate(scaled_mipzy, -1 * flip * angle, 'bilinear');
            figure(2)
            imagesc(rot_scaled_mipzy); axis equal tight

            mask = rot_scaled_mipzy > 0;
            [row, col] = find(mask);
            min_row = min(row); max_row = max(row);
            min_col = min(col); max_col = max(col);

            cropped_mipzy = rot_scaled_mipzy(min_row:max_row, min_col:max_col);
            figure(2)
            imagesc(cropped_mipzy); axis equal tight

            zy_view = permute(scaled_ShearImage, [1, 3, 2]);

            tic
            rotated_zy = single(zeros(size(rot_scaled_mipzy,1), size(rot_scaled_mipzy,2), size(zy_view,3)));

            for i = 1:size(zy_view, 3)
                rotated_zy(:,:,i) = imrotate(zy_view(:,:,i), -1 * flip * angle, 'bilinear');
            end

            cropped_rotate_zy = rotated_zy(min_row:max_row, min_col:max_col, :);
            toc

            rotTop_ShearImage = permute(cropped_rotate_zy, [1, 3, 2]);
            rotTop_ShearImage = uint16(rotTop_ShearImage);

            output_size_rotTop = size(rotTop_ShearImage,1) * size(rotTop_ShearImage,2) * size(rotTop_ShearImage,3) * 2 / (1024*1024*1024);

            outputFolder2 = fullfile(output_dir, strcat('Top_shear', num2str(angle), '_mlv2_', cellNameWithIndex));
            if ~isfolder(outputFolder2)
                mkdir(outputFolder);
            end

            disp("Saving the top-view image");
            if output_size_rotTop > 4
                disp("File is larger than 4GB, saving as BigTIFF format");
            end

            tic
            writetiffstack(rotTop_ShearImage, fullfile(outputFolder2, [baseName '.tif']));

            msg = 'z pixel = x(y) pixel. Full resolution preserved.';
            if output_size_rotTop > 4
                msg = sprintf('%s BigTIFF format used for %.2f GB file.', msg, output_size_rotTop);
            end

            note_filename = fullfile(outputFolder2, 'note.txt');
            fileID = fopen(note_filename, 'w');
            fprintf(fileID, '%s', msg);
            fclose(fileID);
            toc;
            toc;
        end
    end
end

toc
disp("All processing completed successfully.");