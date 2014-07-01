// Get directory environment
[u,t,n]=file();
i = grep(n',"/(?:.*\.sci|.*\.sce)$/","r");
script = n(i(1));
env = pwd();

// Include everything from the source dir
cd(dirname(script));
exec("/usr/share/scilab/contrib/sivp/loader.sce");
exec("transform.sci");
cd(env);

errcatch(-1,"stop");

// Read arguments
args = sciargs();
argv = [];
for i = 1:size(args, 2)
    if args(i) == '-args' & i != length(args);
        argv = args(i+1:$);
    end
end
argc = size(argv, 2);

// Parse arguments (poor man's getopt)
tfunctionals = [];
pfunctionals = [];
angle_interval = 1;
iterations = 0;
inputs = [];
i = 1;
while i <= argc
    arg = argv(i);

    if arg == '-m' | arg == '--mode'
        program_mode = argv(i+1);
        i = i+1;
    elseif arg == '-T' | arg == '--t-functional'
        tfunctionals($+1) = int(strtod(argv(i+1)));
        i = i+1;
    elseif arg == '-P' | arg == '--p-functional'
        pfunctionals($+1) = int(strtod(argv(i+1)));
        i = i+1;
    elseif arg == '-a' | arg == '--angle'
        angle_interval = int(strtod(argv(i+1)));
        i = i+1;
    elseif arg == '-n' | arg == '--iterations'
        iterations = int(strtod(argv(i+1)));
        i = i+1;
    elseif part(arg, 1) == '-'
        printf("Unknown argument %s\n", argv(i));
    else
        inputs($+1) = arg;
    end
    i = i + 1;
end
if size(inputs, 2) <> 1
    error("Please provide a single input image");
end
imageFile = inputs(1);
if program_mode == "benchmark" & iterations == 0
    error("Required argument iterations was not provided")
end

// Check orthonormal
orthonormal_count = 0;
for i=1:length(pfunctionals)
    if pfunctionals(i) > 3
        orthonormal_count = orthonormal_count + 1;
    end
end
if orthonormal_count == 0
    orthonormal = %f;
elseif orthonormal_count == length(pfunctionals)
    orthonormal = %t;
else
    error('cannot mix orthonormal with non-orthonormal pfunctionals')
end

// Gather input data
image = imread(imageFile);
image = mat2gray(image(:, :, 1));    // since scilab reads colour when there is none
[padded base_name] = prepare_transform(image, imageFile, angle_interval, orthonormal);

if program_mode == "calculate"
    // Get output data
    [sinogram circus] = get_transform(padded, tfunctionals, pfunctionals, angle_interval, orthonormal);

    // Save sinograms
    for t_i = 1:length(tfunctionals)
        t = tfunctionals(t_i);

        _trace = sinogram(:, :, t_i);
        csvWrite(_trace, sprintf('%s-T%d.csv', base_name, t));
        // TODO: debug flag
        imwrite(mat2gray(_trace), sprintf('%s-T%d.pgm', base_name, t));
    end

    // Save circus functions
    for t_i = 1:length(tfunctionals)
        t = tfunctionals(t_i);
        for p_i = 1:length(pfunctionals)
            p = pfunctionals(p_i);
            if p >= 4
                p_real = p - 3;
                type = 'H';
            else
                p_real = p;
                type = 'P';
            end

            _trace = circus(:, p_i + (t_i-1)*length(pfunctionals));
            csvWrite(_trace, sprintf('%s-T%d_%s%d.csv', base_name, t, type, p_real));
        end
    end

elseif program_mode == "benchmark"
    // Warm-up
    for i=1:3
        get_transform(padded, tfunctionals, pfunctionals, angle_interval, orthonormal);
    end

    for i=1:iterations
        tic();
        get_transform(padded, tfunctionals, pfunctionals, angle_interval, orthonormal);
        telapsed = toc();
        printf('t_%g=%g\n', i, telapsed);
    end

elseif program_mode == "profile"
    // Warm-up
    get_transform(padded, tfunctionals, pfunctionals, angle_interval, orthonormal);

    // Add profiling instructions
    funcprot(0);
    add_profiling("get_transform")

    get_transform(padded, tfunctionals, pfunctionals, angle_interval, orthonormal);
    profile(get_transform);
    showprofile(get_transform);

else
    error('invalid execution mode')
end
