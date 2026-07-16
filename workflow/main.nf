#!/usr/bin/env nextflow
nextflow.enable.dsl=2

include { STAGE_DECON_INPUT } from './modules'
include { STAGE_DECON_TIFF_INPUT } from './modules'
include { DECON } from './modules'
include { EXPORT_OUTPUT_FORMAT } from './modules'

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

def isTiffInputPattern(inputPattern) {
    def text = inputPattern.toString().trim().toLowerCase()
    return text.endsWith('.tif') || text.endsWith('.tiff')
}

workflow {
    input_patterns = normalizeInputPatterns(params.input, workflow.commandLine)
    if (input_patterns) {
        log.info "Selected ${input_patterns.size()} input image(s): ${input_patterns.join(', ')}"
        input_files_ch = Channel
            .fromList(input_patterns)
            .map { input_pattern -> file(input_pattern, checkIfExists: true) }
            .collect()
        if (input_patterns.every { input_pattern -> isTiffInputPattern(input_pattern) }) {
            log.info "Bypassing OME-Zarr input normalization for TIFF input(s)."
            STAGE_DECON_TIFF_INPUT(input_files_ch)
            decon_input_ch = STAGE_DECON_TIFF_INPUT.out.decon_input_dir
        } else {
            STAGE_DECON_INPUT(input_files_ch)
            decon_input_ch = STAGE_DECON_INPUT.out.decon_input_dir
        }
    } else {
        decon_input_ch = Channel.value(params.image_path)
    }

    DECON(
        decon_input_ch,
        params.background,
        params.iter,
        params.output_dir
    )

    if (params.output_formats == 'tiff') {
        EXPORT_OUTPUT_FORMAT(DECON.out.decon_output, params.output_formats)
    }
}
