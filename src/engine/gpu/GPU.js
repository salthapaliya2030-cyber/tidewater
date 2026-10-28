// The WebGPU device and the per-frame command encoder.
//
// Everything GPU-side goes through this singleton: `GPU.device`, `GPU.queue`, the shared samplers
// and `GPU.encoder` (one command encoder per frame, submitted by `GPU.submit()`). Systems record
// their compute dispatches and render passes into the current encoder in call order.
//
// Ordering note: queue.writeBuffer() calls land before the frame's command buffer runs, so a buffer
// range written twice in one frame keeps only the last value for every pass. Anything that changes
// between passes of a frame needs its own buffer (or its own offset).

function reportGPUError( title, detail ) {

	if ( typeof window !== 'undefined' && typeof window.__showTidewaterError === 'function' ) {

		window.__showTidewaterError( title, detail );

	}

}

export const GPU = {

	device: null,
	queue: null,
	adapter: null,
	context: null,
	canvas: null,
	format: 'bgra8unorm',
	features: new Set(),
	limits: null,
	hasTimestamp: false,
	hasFloat32Filterable: false,
	encoder: null,
	frame: 0,
	samplers: null,
	_submitHooks: [],

	async init( { canvas = null, requiredLimits = {}, headless = false } = {} ) {

		if ( ! navigator.gpu ) {

			throw new Error( 'WebGPU is unavailable in this browser. Update Chrome or Edge and enable hardware acceleration. Tidewater needs a WebGPU-capable graphics device; lowering graphics settings cannot add WebGPU support.' );

		}

		// Some laptops and managed Chromebooks do not return an adapter for the high-performance
		// hint even when a usable integrated GPU exists. Try low-power and default selection too.
		let adapter = null;
		let adapterError = null;
		for ( const options of [ { powerPreference: 'high-performance' }, { powerPreference: 'low-power' }, undefined ] ) {

			try {

				adapter = options ? await navigator.gpu.requestAdapter( options ) : await navigator.gpu.requestAdapter();
				if ( adapter ) break;

			} catch ( e ) {

				adapterError = e;

			}

		}

		if ( ! adapter ) {

			const detail = adapterError && adapterError.message ? ` (${ adapterError.message })` : '';
			throw new Error( 'No WebGPU adapter could be created. Update the browser and graphics driver, and enable hardware acceleration. This device may not support Tidewater.' + detail );

		}

		this.adapter = adapter;

		const L = adapter.limits;
		const want = {
			maxSampledTexturesPerShaderStage: 32,
			maxSamplersPerShaderStage: 16,
			maxStorageBuffersPerShaderStage: 10,
			maxStorageTexturesPerShaderStage: 8,
			maxComputeWorkgroupStorageSize: 32768,
			maxColorAttachmentBytesPerSample: 64,
			maxStorageBuffersInVertexStage: 4,
			maxStorageBuffersInFragmentStage: 8,
			maxStorageTexturesInFragmentStage: 4,
			maxBindingsPerBindGroup: 1000,
			maxBufferSize: 1024 * 1024 * 1024,
			maxStorageBufferBindingSize: 512 * 1024 * 1024,
			...requiredLimits,
		};
		const limits = {};
		for ( const k in want ) if ( L[ k ] !== undefined ) limits[ k ] = Math.min( want[ k ], L[ k ] );

		const optional = [ 'float32-filterable', 'timestamp-query', 'rg11b10ufloat-renderable', 'float32-blendable', 'shader-f16', 'clip-distances' ];
		const requiredFeatures = optional.filter( ( f ) => adapter.features.has( f ) );
		this.features = new Set( requiredFeatures );
		this.hasTimestamp = this.features.has( 'timestamp-query' );
		this.hasFloat32Filterable = this.features.has( 'float32-filterable' );

		const device = await adapter.requestDevice( { requiredFeatures, requiredLimits: limits } );
		this.device = device;
		this.queue = device.queue;
		this.limits = device.limits;
		device.lost.then( ( info ) => {

			const message = 'Reason: ' + ( info.reason || 'unknown' ) + '\n' + ( info.message || 'The GPU device was lost without additional detail.' );
			console.error( 'WebGPU device lost:', message );
			reportGPUError( 'WebGPU device lost', message );

		} );
		device.addEventListener && device.addEventListener( 'uncapturederror', ( e ) => {

			const message = String( e.error && e.error.message || e.message || e.error || e ).split( '\n' ).slice( 0, 10 ).join( '\n' );
			console.error( 'WebGPU uncaptured error:', message );
			reportGPUError( 'WebGPU validation/runtime error', message );

		} );

		if ( canvas && ! headless ) {

			this.canvas = canvas;
			this.context = canvas.getContext( 'webgpu' );
			this.format = navigator.gpu.getPreferredCanvasFormat();
			this.context.configure( { device, format: this.format, alphaMode: 'opaque', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_DST } );

		}

		this._createSamplers();
		return this;

	},

	_createSamplers() {

		const d = this.device;
		const s = ( desc ) => d.createSampler( desc );
		const lin = { magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear' };
		this.samplers = {
			linearRepeat: s( { ...lin, addressModeU: 'repeat', addressModeV: 'repeat', addressModeW: 'repeat' } ),
			linearClamp: s( { ...lin, addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge', addressModeW: 'clamp-to-edge' } ),
			linearMirror: s( { ...lin, addressModeU: 'mirror-repeat', addressModeV: 'mirror-repeat', addressModeW: 'mirror-repeat' } ),
			anisoRepeat: s( { ...lin, addressModeU: 'repeat', addressModeV: 'repeat', addressModeW: 'repeat', maxAnisotropy: 8 } ),
			// 4x: the terrain detail texture (grazing views of the beach; 8x costs ~0.5 ms more at 1440p)
			aniso4Repeat: s( { ...lin, addressModeU: 'repeat', addressModeV: 'repeat', addressModeW: 'repeat', maxAnisotropy: 4 } ),
			anisoClamp: s( { ...lin, addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge', maxAnisotropy: 8 } ),
			nearestClamp: s( { magFilter: 'nearest', minFilter: 'nearest', mipmapFilter: 'nearest', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge', addressModeW: 'clamp-to-edge' } ),
			nearestRepeat: s( { magFilter: 'nearest', minFilter: 'nearest', mipmapFilter: 'nearest', addressModeU: 'repeat', addressModeV: 'repeat', addressModeW: 'repeat' } ),
			shadow: s( { magFilter: 'linear', minFilter: 'linear', compare: 'less', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge' } ),
		};

	},

	// Start recording a frame. Anything recorded before the first beginFrame() also works: the
	// encoder is created on demand.
	beginFrame() {

		this.frame ++;
		return this.getEncoder();

	},

	getEncoder() {

		if ( ! this.encoder ) this.encoder = this.device.createCommandEncoder();
		return this.encoder;

	},

	// Submit everything recorded so far. Hooks run right after the submit (readback mapping).
	submit() {

		if ( ! this.encoder ) return;
		const hooks = this._submitHooks;
		this._submitHooks = [];
		for ( const h of hooks ) if ( h.before ) h.before( this.encoder );
		this.queue.submit( [ this.encoder.finish() ] );
		this.encoder = null;
		for ( const h of hooks ) if ( h.after ) h.after();

	},

	onSubmit( before, after ) {

		this._submitHooks.push( { before, after } );

	},

	// ---- pipelines
	// Pipelines compile asynchronously: the browser compiles them in parallel on worker threads, and a
	// pipeline created synchronously would stall the GPU process (and with it the page) at first use.
	// Handles are { pipeline, label }; `pipeline` is null until the compile finishes. Users that must
	// run this frame (compute, post) call ready( handle ), which falls back to a synchronous create.
	_pending: new Set(),
	_compileQueue: [],
	_activeCompiles: 0,
	// Constrained integrated GPUs can lose their device if dozens of heavy pipelines compile together.
	_enqueueCompile( fn ) {

		return new Promise( ( resolve, reject ) => {

			this._compileQueue.push( { fn, resolve, reject } );
			this._drainCompileQueue();

		} );

	},

	_drainCompileQueue() {

		const limit = ( this.limits?.maxSampledTexturesPerShaderStage ?? 32 ) < 24 ? 1 : 8;
		while ( this._activeCompiles < limit && this._compileQueue.length ) {

			const task = this._compileQueue.shift();
			this._activeCompiles ++;
			Promise.resolve().then( task.fn ).then( task.resolve, task.reject ).finally( () => {

				this._activeCompiles --;
				this._drainCompileQueue();

			} );

		}

	},

	syncCompiles: [], // labels of pipelines needed before their async compile finished (diagnostics)

	renderPipeline( desc ) {

		return this._async( desc, 'render' );

	},

	computePipeline( desc ) {

		return this._async( desc, 'compute' );

	},

	_async( desc, kind ) {

		const h = { pipeline: null, label: desc.label, desc, kind, failed: false };
		// Keep device-heavy work bounded on the WebGPU baseline profile. Requests can be queued
		// together, but only a few pipelines compile at once during startup on integrated GPUs.
		const p = this._enqueueCompile( () => {

			if ( h.pipeline ) return;
			const create = kind === 'render' ? this.device.createRenderPipelineAsync : this.device.createComputePipelineAsync;
			return create.call( this.device, desc ).then( ( pipeline ) => {

				if ( ! h.pipeline ) h.pipeline = pipeline;
				h.desc = null;

			}, ( e ) => {

				h.failed = true;
				h.error = e;
				const message = String( e && ( e.message || e ) ).split( '\n' ).slice( 0, 10 ).join( '\n' );
				console.error( `WebGPU: pipeline "${ desc.label }" failed asynchronously: ${ message }` );
				reportGPUError( `WebGPU pipeline failed: ${ desc.label }`, message );

			} );

		} ).finally( () => this._pending.delete( p ) );
		this._pending.add( p );
		return h;

	},

	// the pipeline now (synchronous compile when the async one has not finished)
	ready( h ) {

		if ( h.pipeline ) return h.pipeline;
		this.syncCompiles.push( h.label );
		try {

			h.pipeline = h.kind === 'render' ? this.device.createRenderPipeline( h.desc ) : this.device.createComputePipeline( h.desc );
			h.failed = false;
			h.desc = null;
			return h.pipeline;

		} catch ( e ) {

			const asyncDetail = h.error && h.error.message ? ` Async compilation: ${ h.error.message.split( '\n' ).slice( 0, 4 ).join( '\n' ) }` : '';
			throw new Error( `WebGPU pipeline "${ h.label }" failed to compile: ${ e.message.split( '\n' ).slice( 0, 8 ).join( '\n' ) }.${ asyncDetail }` );

		}

	},

	// resolves when every pipeline requested so far has compiled
	async pipelinesReady() {

		while ( this._pending.size ) await Promise.all( [ ...this._pending ] );

	},

	// A compute or render pass on the frame encoder.
	computePass( label, fn, timestampWrites ) {

		const pass = this.getEncoder().beginComputePass( { label, timestampWrites } );
		fn( pass );
		pass.end();

	},

};

// Texture format helpers ------------------------------------------------------------------------

const FORMAT_INFO = {
	r8unorm: { bytes: 1, sample: 'float' },
	rg8unorm: { bytes: 2, sample: 'float' },
	rgba8unorm: { bytes: 4, sample: 'float' },
	'rgba8unorm-srgb': { bytes: 4, sample: 'float' },
	bgra8unorm: { bytes: 4, sample: 'float' },
	r16float: { bytes: 2, sample: 'float' },
	rg16float: { bytes: 4, sample: 'float' },
	rgba16float: { bytes: 8, sample: 'float' },
	rg11b10ufloat: { bytes: 4, sample: 'float' },
	r32float: { bytes: 4, sample: 'unfilterable-float' },
	rg32float: { bytes: 8, sample: 'unfilterable-float' },
	rgba32float: { bytes: 16, sample: 'unfilterable-float' },
	r32uint: { bytes: 4, sample: 'uint' },
	rg32uint: { bytes: 8, sample: 'uint' },
	rgba32uint: { bytes: 16, sample: 'uint' },
	r32sint: { bytes: 4, sample: 'sint' },
	r8uint: { bytes: 1, sample: 'uint' },
	rgba8uint: { bytes: 4, sample: 'uint' },
	depth32float: { bytes: 4, sample: 'depth' },
	depth24plus: { bytes: 4, sample: 'depth' },
	'depth24plus-stencil8': { bytes: 4, sample: 'depth' },
};

export function formatInfo( format ) {

	const i = FORMAT_INFO[ format ];
	if ( ! i ) throw new Error( 'Unknown texture format ' + format );
	return i;

}

// sample type of a format for bind group layouts (32-bit float formats are filterable with the feature)
export function sampleTypeOf( format ) {

	const t = formatInfo( format ).sample;
	if ( t === 'unfilterable-float' && GPU.hasFloat32Filterable ) return 'float';
	return t;

}
