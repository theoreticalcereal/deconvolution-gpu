%% Configuration Parameters

% =========================================================================
% INJECTED BY NEXTFLOW / PYTHON WRAPPER
% The following variables are passed directly into the workspace:
% imagePath, psfPath, psfFile, background, iter
% DO NOT hardcode them here, or it will break the pipeline!
% =========================================================================

% Map the string 'psfFile' passed by Python to the cell array 'psf' expected by the script
if exist('psfFile', 'var')
    psf{1} = psfFile;
end

% Set defaults for variables that might not be passed by the wrapper yet
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


%% load PSF
for p=1: size(psf,2)
    filepath=fullfile(psfPath,psf{p});
    if ~isfile(filepath)
        error('Missing expected file: %s', filepath);
    end
    PSFimage=readtiffstack(filepath);
  
PSFimage=double(PSFimage);
PSFimage=PSFimage-background;
PSFimage=abs(PSFimage);
%PSFimage=PSFimage./sum(sum(sum(PSFimage)));
PSF{p}=PSFimage;

end
clear PSFimage

%% Deconvolution
%names1 = dir(fullfile(imagePath,'Cell*'));
%names1 = dir(fullfile(imagePath,'Shear_Cell*_45'));
     
%if size(Cell_index)==0
%    numfolder=size(names1,1);
%else
%    numfolder=size(Cell_index,2);
%end

numfolder=size(Cell_index,2);
ch_number= size(ChannelstoProcess,2);

for c=1:numfolder
    
    names2=strcat(Cell_name,num2str(Cell_index(c)));

    % Determine input files robustly    
    files = dir(fullfile(imagePath, names2, '*.tiff'));
    if isempty(files)
        files = dir(fullfile(imagePath, names2, '*.tif'));
    end
    if isempty(files)
        error('No TIFF files found in %s', fullfile(imagePath, names2));
    end
    numImages = numel(files);
    if size(timepoint,2)==0
        t_st=0;
        %t_end=(size(names2,1)-2)/ch_number;
        t_end=round(numImages/ch_number)-1;
    else
        t_st=min(timepoint);
        t_end=max(timepoint);
    end
    
    for t=t_st:t_end
    
        for ch=1:ch_number
        
        tic
       
       baseName = sprintf('CH%02d_%06d_registered_consistent', ChannelstoProcess(ch), t);

        candidates = {
            fullfile(imagePath, names2, [baseName '.tif'])
            fullfile(imagePath, names2, [baseName '.tiff'])
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
%       InfoImage=imfinfo(filepath);
%       mImage=InfoImage(1).Height;
%       nImage=InfoImage(1).Width;
%       NumberImages=length(InfoImage);
%       %NumberImages=958;
%       
%       FinalImage=zeros(mImage,nImage,NumberImages,'uint16');
%
%       TifLink = Tiff(filepath, 'r');
%for i=1:NumberImages
%   TifLink.setDirectory(i);
%   FinalImage(:,:,i)=TifLink.read();
%end
%TifLink.close();

FinalImage= readtiffstack(filepath);
mImage= size(FinalImage,1);
nImage= size(FinalImage,2);
NumberImages= size(FinalImage,3);
%% Deconvolution
%padx=ceil(mImage/2);
%pady=ceil(nImage/2);
%padz=ceil(NumberImages/2);
%E1=padarray(single(FinalImage),[20 20],'symmetric') ; 
E1=padarray(single(FinalImage),[20 20 20],'symmetric') ; 
maxE1=max(E1(:));
minE1=min(E1(:));
psfi=single(PSF{ch});
%psfi=psfi./max(psfi(:));
[Dec,psfr]=deconvblind(E1,psfi,iter);
%Dec=Dec(21:20+mImage,21:20+nImage);
Dec=Dec(21:20+mImage,21:20+nImage,21:20+NumberImages);
Dec = (Dec - min(Dec(:))) / (max(Dec(:)) - min(Dec(:)));
Dec=Dec.*(maxE1-minE1)+minE1;
%Dec=Dec./max(Dec(:));
Dec=uint16(Dec);

%% save deconvolved image
%finalPath=fullfile(dir_Dec,strcat('FirstBlind_',names2));
%mkdir(finalPath);

%Decname=fullfile(finalPath,strcat('Dec_',filename));
%Decname=fullfile(finalPath,filename);
    
%    [nx, ny, nz]= size(Dec);
%    imgType= class(Dec);
%    tagstruct.Photometric= Tiff.Photometric.MinIsBlack;
%    tagstruct.ImageLength = nx;
%    tagstruct.ImageWidth = ny;
%    tagstruct.PlanarConfiguration= Tiff.PlanarConfiguration.Chunky;
%    tagstruct.Compression = Tiff.Compression.None;
%    tagstruct.BitsPerSample= 16;
    
%    tiffFile=Tiff(Decname, 'w');
    
%    for iz=1:nz
%        tiffFile.setTag(tagstruct);
%        tiffFile.write(Dec(:,:,iz));
%        tiffFile.writeDirectory();
        
%    end
%    tiffFile.close();
    
%% save retrived psf safely
mx = max(psfr(:));
if mx == 0
    error('Estimated PSF is all zeros.');
end
psfr = psfr ./ mx;
psfr2=uint16(60000*psfr);

PSFfolder=fullfile(dir_Dec,strcat('PSFr_',names2));
mkdir(PSFfolder);
PSFname=fullfile(PSFfolder,strcat(names2,'psfr',num2str(ch),'.tif'));

writetiffstack(psfr2, PSFname);
    
%    [nx, ny, nz]= size(psfr2);
%    imgType= class(psfr2);
%    tagstruct.Photometric= Tiff.Photometric.MinIsBlack;
%    tagstruct.ImageLength = nx;
%    tagstruct.ImageWidth = ny;
%    tagstruct.PlanarConfiguration= Tiff.PlanarConfiguration.Chunky;
%    tagstruct.Compression = Tiff.Compression.None;
%    tagstruct.BitsPerSample= 16;
    
%    tiffFile=Tiff(PSFname, 'w');
    
%    for iz=1:nz
%        tiffFile.setTag(tagstruct);
%        tiffFile.write(psfr2(:,:,iz));
%        tiffFile.writeDirectory();
        
%    end
%    tiffFile.close();


%% DoubleBlindDeconvolution
%[Dec2]=deconvwnr(E1,psfr,100);
[Dec2]=deconvlucy(E1,psfr,iter);
%[Dec2]=deconvreg(E1,psfr);
%[Dec2,psfr3]=deconvblind(E1,psfr,iter);
%Dec2=Dec2(21:20+mImage,21:20+nImage);
Dec2=Dec2(21:20+mImage,21:20+nImage,21:20+NumberImages);
Dec2 = (Dec2 - min(Dec2(:))) / (max(Dec2(:)) - min(Dec2(:)));
Dec2=Dec2.*(maxE1-minE1)+minE1;
%Dec=Dec./max(Dec(:));
Dec2=uint16(Dec2);
%Dec2=Dec2./max(Dec2(:));
%Dec2=uint16(Dec2*maxE1);

%% save DoubleBlindDeconvolved image
%finalPath2=fullfile(dir_Dec,strcat('Tikhonov_',names2));
%finalPath2=fullfile(dir_Dec,strcat('Wiener0.1_',names2));
finalPath2=fullfile(dir_Dec,strcat('DB2_',names2));
mkdir(finalPath2);
Decname2=fullfile(finalPath2,filename);
    
       writetiffstack(Dec2,Decname2);
    %writeBigtiffstack(Dec2,Decname2);
%% save retrived psf 
%psfr3=psfr3./max(psfr3(:));
%psfr4=uint16(60000*psfr3);

%PSFname2=fullfile(dir_Dec,strcat(names2,'DBpsfr',num2str(c),'.tif'));
    
%    [nx, ny, nz]= size(psfr4);
%    imgType= class(psfr4);
%    tagstruct.Photometric= Tiff.Photometric.MinIsBlack;
%    tagstruct.ImageLength = nx;
%    tagstruct.ImageWidth = ny;
%    tagstruct.PlanarConfiguration= Tiff.PlanarConfiguration.Chunky;
%    tagstruct.Compression = Tiff.Compression.None;
%    tagstruct.BitsPerSample= 16;
   
%    tiffFile=Tiff(PSFname2, 'w');
    
%    for iz=1:nz
%        tiffFile.setTag(tagstruct);
%        tiffFile.write(psfr4(:,:,iz));
%        tiffFile.writeDirectory();
%        
%    end
%    tiffFile.close();
         
    
toc, disp('Done')
        end
    end
end

disp('All Done')