#!/usr/bin/env nextflow
nextflow.enable.dsl=2

include { DESKEW } from './modules'
include { BUILD_DECON_CONTAINER } from './modules'
include { STAGE_DECON_INPUT } from './modules'
include { DECON }  from './modules'

def inputTextFromCommandLine(commandLine) {
    if (!commandLine) {
        return null
    }

    def matcher = (commandLine =~ /(?:^|\s)--input\s+(.+?)(?=\s+--[A-Za-z0-9_][A-Za-z0-9_-]*\b|$)/)
    return matcher.find() ? matcher.group(1).trim() : null
}

def normalizeInputPatterns(input, commandLine = null) {
    if (!input) {
        return []
    }

    def input_text = (input instanceof List)
        ? input.collect { it.toString() }.join(',')
        : input.toString()

    input_text = input_text.trim()
    def command_input_text = inputTextFromCommandLine(commandLine)
    if (command_input_text && (
            input_text.count('[') != input_text.count(']') ||
            command_input_text.startsWith('[') ||
            command_input_text.contains(input_text))) {
        input_text = command_input_text
    }

    if (input_text.startsWith('[') && input_text.endsWith(']')) {
        input_text = input_text.substring(1, input_text.length() - 1).trim()
    }

    return input_text
        .split(/\s*,\s*/)
        .collect { input_pattern ->
            input_pattern
                .trim()
                .replaceAll(/^[\['"\s]+/, '')
                .replaceAll(/[\]'"\s]+$/, '')
        }
        .findAll { it }
}

workflow {
    input_patterns = normalizeInputPatterns(params.input, workflow.commandLine)
    if (input_patterns) {
        log.info "Selected ${input_patterns.size()} input TIFF(s): ${input_patterns.join(', ')}"
        input_tiffs_ch = Channel
            .fromList(input_patterns)
            .map { input_pattern -> file(input_pattern, checkIfExists: true) }
            .collect()
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
        if (input_patterns) {
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
        deskew_image_path_ch = input_patterns ? selected_input_dir_ch : Channel.value(params.image_path)
        deskew_cell_name = input_patterns ? '' : params.cell_name

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
