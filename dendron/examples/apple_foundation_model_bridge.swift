#!/usr/bin/env swift
//
// apple_foundation_model_bridge.swift
// smart-agent-tool
//
// Lightweight CLI bridge to Apple Foundation Models framework (macOS 26.0+).
// Reads JSON request from stdin:
//   { "instructions": "<system prompt>", "prompt": "<user prompt>" }
// And outputs JSON response:
//   { "status": "success", "content": "<response text>" }
//

import Foundation
#if canImport(FoundationModels)
import FoundationModels
#endif

struct RequestPayload: Codable {
    let instructions: String?
    let prompt: String
}

struct ResponsePayload: Codable {
    let status: String
    let content: String?
    let error: String?
}

func sendResponse(_ payload: ResponsePayload) {
    let encoder = JSONEncoder()
    encoder.outputFormatting = .withoutEscapingSlashes
    if let data = try? encoder.encode(payload),
       let jsonString = String(data: data, encoding: .utf8) {
        print(jsonString)
    } else {
        print("{\"status\":\"error\",\"error\":\"Failed to encode response\"}")
    }
}

// Read standard input
let inputData = FileHandle.standardInput.readDataToEndOfFile()
guard !inputData.isEmpty else {
    sendResponse(ResponsePayload(status: "error", content: nil, error: "Empty input from stdin."))
    exit(1)
}

let decoder = JSONDecoder()
guard let request = try? decoder.decode(RequestPayload.self, from: inputData) else {
    sendResponse(ResponsePayload(status: "error", content: nil, error: "Invalid JSON input format."))
    exit(1)
}

#if canImport(FoundationModels)
if #available(macOS 26.0, *) {
    let model = SystemLanguageModel.default
    switch model.availability {
    case .available:
        let session: LanguageModelSession
        if let instructions = request.instructions, !instructions.isEmpty {
            session = LanguageModelSession(model: model, instructions: instructions)
        } else {
            session = LanguageModelSession(model: model)
        }

        do {
            let response = try await session.respond(to: request.prompt)
            sendResponse(ResponsePayload(status: "success", content: response.content, error: nil))
            exit(0)
        } catch {
            sendResponse(ResponsePayload(status: "error", content: nil, error: "Inference failed: \(error.localizedDescription)"))
            exit(1)
        }

    case .unavailable(let reason):
        sendResponse(ResponsePayload(status: "error", content: nil, error: "Apple Foundation Models unavailable: \(reason)"))
        exit(1)
    @unknown default:
        sendResponse(ResponsePayload(status: "error", content: nil, error: "Apple Foundation Models status unknown."))
        exit(1)
    }
} else {
    sendResponse(ResponsePayload(status: "error", content: nil, error: "Apple Foundation Models requires macOS 26.0 or later."))
    exit(1)
}
#else
sendResponse(ResponsePayload(status: "error", content: nil, error: "FoundationModels framework is not available on this system."))
exit(1)
#endif
