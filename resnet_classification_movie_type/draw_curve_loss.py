import matplotlib.pyplot as plt
import matplotlib
losses = [0.7276, 0.4118, 0.2788, 0.2638, 0.2610, 0.2554, 0.2497, 0.2474
    , 0.2469, 0.2453, 0.2434, 0.2423, 0.2420, 0.2413, 0.2400, 0.2392, 0.2391
    , 0.2368, 0.2375, 0.2369, 0.2368, 0.2363, 0.2354, 0.2347, 0.2348, 0.2343
    , 0.2334, 0.2331, 0.2330, 0.2327, 0.2319, 0.2312, 0.2308, 0.2302, 0.2298
    , 0.2296, 0.2293, 0.2288, 0.2283, 0.2283, 0.2281, 0.2276, 0.2271, 0.2270
    , 0.2267, 0.2260, 0.2259, 0.2257, 0.2254, 0.2251, 0.2244, 0.2246, 0.2243
    , 0.2238, 0.2234, 0.2232, 0.2230, 0.2228, 0.2225, 0.2223, 0.2221, 0.2218
    , 0.2215, 0.2215, 0.2213, 0.2206, 0.2207, 0.2205, 0.2202, 0.2195, 0.2195
    , 0.2194, 0.2189, 0.2188, 0.2189, 0.2185, 0.2181, 0.2182, 0.2178, 0.2175]

plt.plot(losses)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.savefig("mygraph.png")
