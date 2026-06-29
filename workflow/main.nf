#!/usr/bin/env nextflow
nextflow.enable.dsl=2

include { DESKEW } from './modules'
include { BUILD_DECON_CONTAINER } from './modules'
include { STAGE_DECON_INPUT } from './modules'
include { DECON }  from './modules'
include { CONVERT_TIFFS_TO_NEUROGLANCER } from './modules'

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

def isSupplied(value) {
    if (value == null) {
        return false
    }
    def text = value.toString().trim()
    return text && text != '-1' && text != '-1.0'
}

def optionalValue(value) {
    return isSupplied(value) ? value : ''
}

def requireSupplied(name, value, context) {
    if (!isSupplied(value)) {
        throw new IllegalArgumentException("${name} must be provided for ${context}; -1 means unset and is ignored only when that parameter is optional.")
    }
    return value
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

    BUILD_DECON_CONTAINER()
    decon_container_ch = BUILD_DECON_CONTAINER.out.image

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
        CONVERT_TIFFS_TO_NEUROGLANCER(DECON.out.decon_output, decon_container_ch)
    } else {
        deskew_image_path_ch = input_patterns ? selected_input_dir_ch : Channel.value(params.image_path)
        deskew_cell_name = input_patterns ? '' : params.cell_name
        deskew_cell_index = optionalValue(params.cell_index)
        deskew_dx = requireSupplied('dx', params.dx, 'deskew runs')
        deskew_dz = requireSupplied('dz', params.dz, 'deskew runs')

        DESKEW(
            deskew_image_path_ch,
            deskew_cell_name,
            deskew_cell_index,
            deskew_dx,
            deskew_dz,
            params.angle,
            params.flip,
            params.output_dir,
            decon_container_ch
        )

        DECON(
            DESKEW.out.deskewed_path,
            params.background,
            params.iter,
            params.output_dir,
            decon_container_ch
        )
        CONVERT_TIFFS_TO_NEUROGLANCER(DECON.out.decon_output, decon_container_ch)
    }
}
