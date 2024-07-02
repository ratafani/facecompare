//
//  ContentView.swift
//  facecompare
//
//  Created by Muhammad Tafani Rabbani on 01/07/24.
//

import SwiftUI

struct ContentView: View {
    @StateObject var vm = ViewModel()
    
    var body: some View {
        VStack {
            if let image = vm.images[vm.index1]{
                Image(uiImage: image)
                    .resizable()
                    .frame(width: 300,height: 300)
                    .onTapGesture {
                        vm.change(num: 0)
                    }
            }
            
            if let image = vm.images[vm.index2]{
                Image(uiImage: image)
                    .resizable()
                    .frame(width: 300,height: 300)
                    .onTapGesture {
                        vm.change(num: 1)
                    }
            }
            
            Button("predict"){
                vm.predict()
            }
            
            if let res = vm.result{
                Text("\(res)")
            }
            
        }
        .background((vm.result ?? false ? .green : .red ) )
        .ignoresSafeArea()
    }
}

import CoreML

class ViewModel : ObservableObject{
    
    
    @Published var images = [UIImage(named: "real"),UIImage(named: "real2"),UIImage(named: "robot"),UIImage(named: "robot2"),UIImage(named: "real3"),UIImage(named: "real4"),UIImage(named: "real5"),UIImage(named: "real6"),UIImage(named: "real7"),UIImage(named: "real8"),UIImage(named: "real9"),UIImage(named: "real10"),UIImage(named: "real11")]
    @Published var index1 = 0
    @Published var index2 = 1
    
    @Published var result : Bool? = nil
    let service = FaceService()
    
    
    func change(num:Int){
        if num == 0{
            if index1 == 12 {
                index1 = 0
            }else{
                index1+=1
            }
            
            
        }else{
            if index2 == 12 {
                index2 = 0
            }else{
                index2+=1
            }
        }
    }
    
    func predict(){
        
        if let img1 = images[index1],let img2 = images[index2] {
            
            let res = service.compareFaces(image1: img1, image2: img2)
            print(res)
            result = res
        }
        
    }
    
}

class FaceService{
    // Function to resize UIImage to the specified size
    func resizeImageTo160x160(image: UIImage) -> UIImage? {
        let targetSize = CGSize(width: 160, height: 160)
        
        // Create a graphics context and draw the resized image
        UIGraphicsBeginImageContextWithOptions(targetSize, false, 1.0)
        image.draw(in: CGRect(origin: .zero, size: targetSize))
        let resizedImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        
        return resizedImage
    }
    
    // Function to convert UIImage to CVPixelBuffer
    func pixelBuffer(from image: UIImage) -> CVPixelBuffer? {
        guard let cgImage = image.cgImage else {
            return nil
        }
        
        let width = cgImage.width
        let height = cgImage.height
        
        var pixelBuffer: CVPixelBuffer?
        let attributes: [NSObject: AnyObject] = [
            kCVPixelBufferCGImageCompatibilityKey: true as AnyObject,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true as AnyObject
        ]
        
        CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32ARGB, attributes as CFDictionary, &pixelBuffer)
        guard let buffer = pixelBuffer else {
            return nil
        }
        
        CVPixelBufferLockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0))
        let pixelData = CVPixelBufferGetBaseAddress(buffer)
        
        let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
        let context = CGContext(data: pixelData, width: width, height: height, bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(buffer), space: rgbColorSpace, bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue)
        
        guard let drawContext = context else {
            return nil
        }
        
        drawContext.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        CVPixelBufferUnlockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0))
        
        return buffer
    }
    
    // Function to calculate Euclidean distance between two MLMultiArray instances
    func euclideanDistance(_ array1: MLMultiArray, _ array2: MLMultiArray) -> Double {
        
        var sum: Double = 0.0
        for i in 0..<array1.count {
            let diff = array1[i].doubleValue - array2[i].doubleValue
            sum += diff * diff
        }
        return sqrt(sum)
        
        
    }
    
    func normalize(_ array: MLMultiArray) -> MLMultiArray {
        var sum: Double = 0.0
        for i in 0..<array.count {
            sum += array[i].doubleValue * array[i].doubleValue
        }
        
        let length = sqrt(sum)
        for i in 0..<array.count {
            array[i] = NSNumber(value: array[i].doubleValue / length)
        }
        
        return array
    }
    
    func prewhiten(image: UIImage) -> UIImage? {
        guard let cgImage = image.cgImage else { return nil }
        
        let width = cgImage.width
        let height = cgImage.height
        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * width
        let bitsPerComponent = 8
        let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
        
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
              let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: bitsPerComponent, bytesPerRow: bytesPerRow, space: colorSpace, bitmapInfo: bitmapInfo),
              let pixelBuffer = context.data else {
            return nil
        }
        
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        let data = pixelBuffer.bindMemory(to: UInt8.self, capacity: width * height * bytesPerPixel)
        
        var sum: Float = 0.0
        var squaredSum: Float = 0.0
        
        for i in 0..<width * height {
            let r = Float(data[i * 4 + 0])
            let g = Float(data[i * 4 + 1])
            let b = Float(data[i * 4 + 2])
            
            let mean = (r + g + b) / 3.0
            
            data[i * 4 + 0] = UInt8(mean)
            data[i * 4 + 1] = UInt8(mean)
            data[i * 4 + 2] = UInt8(mean)
            
            sum += mean
            squaredSum += mean * mean
        }
        
        let mean = sum / Float(width * height)
        let variance = squaredSum / Float(width * height) - mean * mean
        let stdDev = sqrt(variance)
        
        for i in 0..<width * height {
            let normalizedValue = (Float(data[i * 4 + 0]) - mean) / stdDev
            let clampedValue = min(max(normalizedValue, 0), 255)
            data[i * 4 + 0] = UInt8(clampedValue)
            data[i * 4 + 1] = UInt8(clampedValue)
            data[i * 4 + 2] = UInt8(clampedValue)
        }
        
        let outputCGImage = context.makeImage()
        return outputCGImage.flatMap { UIImage(cgImage: $0) }
    }
    
    // Function to compare two images
    func compareFaces(image1: UIImage, image2: UIImage) -> Bool {
        guard let model = try? facenet(configuration: MLModelConfiguration()) else {
            fatalError("Could not load model")
        }
        
        guard let prewhitenedImage1 = prewhiten(image: image1),
              let prewhitenedImage2 = prewhiten(image: image2) else {
            fatalError("Could not prewhiten images")
        }
        
        guard let resizedImage1 = resizeImageTo160x160(image: prewhitenedImage1),
              let resizedImage2 = resizeImageTo160x160(image: prewhitenedImage2) else {
            fatalError("Could not resize images")
        }
        
        
        guard let buffer1 = pixelBuffer(from: resizedImage1), let buffer2 = pixelBuffer(from: resizedImage2) else {
            fatalError("Could not create pixel buffers")
        }
        
        //        do{
        //            let output1 = try? model.prediction(input: facenetInput(input__0: buffer1))
        //        }catch{
        //
        //        }
        
        
        guard let output1 = try? model.prediction(input__0: buffer1), let output2 = try? model.prediction(input__0: buffer2) else {
            fatalError("Could not make predictions")
        }
        
        
        
        let normalizedOutput1 = normalize(output1.output__0)
        let normalizedOutput2 = normalize(output2.output__0)
        
        
        let distance = euclideanDistance(normalizedOutput1, normalizedOutput2)
        
        // Define a threshold for determining if faces are the same
        let threshold: Double = 0.8
        print(distance,threshold)
        return distance < threshold
    }
}


#Preview {
    ContentView()
}
