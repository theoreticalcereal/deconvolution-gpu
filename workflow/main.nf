#!/usr/bin/env nextflow
nextflow.enable.dsl=2

include { DESKEW } from './modules'
include { BUILD_DECON_CONTAINER } from './modules'
include { STAGE_DECON_INPUT } from './modules'
include { DECON }  from './modules'

def commandLineParam(commandLine, paramName) {
    if (!commandLine) {
        return null
    }

    def matcher = (commandLine =~ /(?:^|\s)--${paramName}\s+(.+?)(?=\s+--[A-Za-z0-9_][A-Za-z0-9_-]*\b|$)/)
    return matcher.find() ? matcher.group(1).trim() : null
}

def normalizePathParam(value, commandLine, paramNames) {
    if (!value) {
        return []
    }

    def names = (paramNames instanceof List) ? paramNames : [paramNames]
    def text = (value instanceof List)
        ? value.collect { it.toString() }.join(',')
        : value.toString()

    for (paramName in names) {
        def commandLineText = commandLineParam(commandLine, paramName)
        if (commandLineText && (commandLineText.startsWith('[') || commandLineText.contains(','))) {
            text = commandLineText
            break
        }
    }

    text = text.trim()
    if (text.startsWith('[') && text.endsWith(']')) {
        text = text.substring(1, text.length() - 1)
    }

    return text
        .split(/\s*,\s*/)
        .collect { pathText ->
            pathText
                .trim()
                .replaceAll(/^[\s\['"]+/, '')
                .replaceAll(/[\s\]'"]+$/, '')
        }
        .findAll { it }
}

workflow {
    input_paths = normalizePathParam(params.image_files ?: params.input, workflow.commandLine, ['image_files', 'input'])
    if (input_paths) {
        log.info "Selected input TIFF(s): ${input_paths.join(', ')}"
        input_tiffs_ch = Channel.fromPath(input_paths, checkIfExists: true).collect()
        STAGE_DECON_INPUT(input_tiffs_ch)
        selected_input_dir_ch = STAGE_DECON_INPUT.out.decon_input_dir
    } else {
        selected_input_dir_ch = Channel.empty()
    }

    if (params.build_decon_container) {
        BUILD_DECON_CONTAINER()
        decon_container_ch = BUILD_DECON_CONTAINER.out.image
    } else {
        decon_container_ch = Channel.value("${projectDir}/images/decon_env.sif")
    }

    if (params.decon_only) {
        if (input_paths) {
            decon_input_ch = selected_input_dir_ch
        } else {
            decon_input_ch = Channel.value(params.decon_input_dir ?: params.image_path)
        }

        DECON(
            decon_input_ch,
            params.background,
            params.iter,
            params.output_dir,
            decon_container_ch
        )
    } else {
        deskew_image_path_ch = input_paths ? selected_input_dir_ch : Channel.value(params.image_path)
        deskew_cell_name = input_paths ? '' : params.cell_name

        DESKEW(
            deskew_image_path_ch,
            deskew_cell_name,
            params.cell_index,
            params.dx,
            params.dz,
            params.angle,
            params.flip,
            params.output_dir
        )

        DECON(
            DESKEW.out.deskewed_path,
            params.background,
            params.iter,
            params.output_dir,
            decon_container_ch
        )
    }
}
