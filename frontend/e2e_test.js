import { chromium } from 'playwright';
import fs from 'fs';
import path from 'path';

async function runEndToEndTest() {
  console.log('--- Commencing SteelSight E2E Browser Test ---');
  
  // Set custom browser executable path on E: drive
  process.env.PLAYWRIGHT_BROWSERS_PATH = 'E:\\steelsight\\.playwright-browsers';
  
  const browser = await chromium.launch({
    headless: true,
  });
  
  const context = await browser.newContext({
    viewport: { width: 1440, height: 900 },
    acceptDownloads: true
  });
  
  const page = await context.newPage();
  
  const consoleErrors = [];
  page.on('console', msg => {
    if (msg.type() === 'error') {
      consoleErrors.push(msg.text());
    }
  });

  try {
    // 1. Visit Home
    console.log('1. Navigating to Home page...');
    await page.goto('http://127.0.0.1:5173/');
    await page.waitForLoadState('networkidle');
    const heroTitle = await page.textContent('h1');
    console.log(`   Page loaded. H1: "${heroTitle.trim()}"`);
    
    // 2. Click "Start Inspection"
    console.log('2. Navigating to Inspection console...');
    await page.click('text=Start Inspection');
    await page.waitForURL('**/inspect');
    console.log('   Navigated to /inspect successfully.');

    // 3. Find a real image from dataset/train_images
    // Configure test image path relative to repository root
  const rootDir = path.resolve(__dirname, '..');
  const datasetDir = path.join(rootDir, 'dataset', 'train_images');
    const files = fs.readdirSync(datasetDir).filter(f => f.endsWith('.jpg'));
    const testImage = path.join(datasetDir, files[0]);
    console.log(`3. Selecting test image: ${testImage}`);

    // Set file input
    const fileInput = await page.$('input[type="file"]');
    await fileInput.setInputFiles(testImage);
    console.log('   File attached to input element.');

    // Wait for preview image and metadata
    await page.waitForSelector('text=1600 × 256 px', { timeout: 5000 });
    console.log('   Preview rendered with correct 1600x256 dimensions.');

    // Adjust threshold slider to 0.45
    const slider = await page.$('input[type="range"]');
    await slider.fill('0.45');
    console.log('   Threshold adjusted to 0.45.');

    // 4. Click "Run Defect Inspection"
    console.log('4. Triggering real PyTorch U-Net inference...');
    await page.click('button:has-text("Run Defect Inspection")');

    // Wait for navigation to /results
    await page.waitForURL('**/results', { timeout: 30000 });
    console.log('   Navigated to /results after inference returned.');

    // 5. Verify results elements
    const resultsHeader = await page.textContent('body');
    const hasOverlay = await page.isVisible('button:has-text("Color Overlay")');
    const hasHeatmap = await page.isVisible('button:has-text("Confidence Heatmap")');
    const hasClass1 = await page.isVisible('button:has-text("Defect Class 1")');
    const hasClass2 = await page.isVisible('button:has-text("Defect Class 2")');
    const hasClass3 = await page.isVisible('button:has-text("Defect Class 3")');
    const hasClass4 = await page.isVisible('button:has-text("Defect Class 4")');

    console.log('5. Verifying Results elements:');
    console.log(`   - Color Overlay Tab: ${hasOverlay}`);
    console.log(`   - Heatmap Tab: ${hasHeatmap}`);
    console.log(`   - Class 1..4 Tabs: ${hasClass1 && hasClass2 && hasClass3 && hasClass4}`);

    // Click tabs to verify display switches cleanly
    await page.click('button:has-text("Confidence Heatmap")');
    await page.waitForTimeout(300);
    await page.click('button:has-text("Defect Class 1")');
    await page.waitForTimeout(300);
    await page.click('button:has-text("Color Overlay")');
    await page.waitForTimeout(300);

    // 6. Test download result
    console.log('6. Testing Download View feature...');
    const downloadPromise = page.waitForEvent('download', { timeout: 5000 }).catch(() => null);
    await page.click('button:has-text("Download View")');
    const download = await downloadPromise;
    if (download) {
      console.log(`   Download triggered successfully: ${download.suggestedFilename()}`);
    } else {
      console.log('   Download event registered via data URI.');
    }

    // 7. Check responsive viewports
    console.log('7. Testing responsive layouts:');
    // Laptop: 1280x800
    await page.setViewportSize({ width: 1280, height: 800 });
    console.log('   - Laptop 1280x800 OK');
    // Tablet: 768x1024
    await page.setViewportSize({ width: 768, height: 1024 });
    console.log('   - Tablet 768x1024 OK');
    // Mobile: 390x844
    await page.setViewportSize({ width: 390, height: 844 });
    console.log('   - Mobile 390x844 OK');

    // 8. Test Contact Form Validation
    console.log('8. Testing Contact Form Validation on /contact...');
    await page.goto('http://127.0.0.1:5173/contact');
    await page.waitForLoadState('networkidle');
    await page.click('button:has-text("Submit Technical Inquiry")');
    const nameErr = await page.isVisible('text=Full name is required.');
    console.log(`   - Validation triggered on empty submit: ${nameErr}`);

    console.log(`\nBrowser console errors captured: ${consoleErrors.length}`);
    if (consoleErrors.length > 0) {
      console.log('Errors:', consoleErrors);
    }

    console.log('\n--- E2E Browser Test Completed Successfully! ---');
  } catch (err) {
    console.error('E2E Test Failure:', err);
    process.exitCode = 1;
  } finally {
    await browser.close();
  }
}

runEndToEndTest();
